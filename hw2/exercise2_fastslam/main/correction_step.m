function particles = correction_step(particles, z)

% Weight the particles according to the current map of the particle
% and the landmark observations z.
% z: struct array containing the landmark observations.
% Each observation z(j) has an id z(j).id, a range z(j).range, and a bearing z(j).bearing
% The vector observedLandmarks indicates which landmarks have been observed
% at some point by the robot.

% Number of particles
numParticles = length(particles);

% Number of measurements in this time step
m = size(z, 2);

% Construct the sensor noise matrix Q_t (2 x 2)
Q_t = 0.1*eye(2);

% process each particle
for i = 1:numParticles
  robot = particles(i).pose;
  % process each measurement
  for j = 1:m
    % Get the id of the landmark corresponding to the j-th observation
    % particles(i).landmarks(l) is the EKF for this landmark
    l = z(j).id;

    % The (2x2) EKF of the landmark is given by
    % its mean particles(i).landmarks(l).mu
    % and by its covariance particles(i).landmarks(l).sigma

    % If the landmark is observed for the first time:
    if (particles(i).landmarks(l).observed == false)

      % Initialize its position based on the measurement and the current robot pose
      % Convert from polar to Cartesian coordinates
      range = z(j).range;
      bearing = z(j).bearing;
      x = robot(1) + range * cos(bearing + robot(3));
      y = robot(2) + range * sin(bearing + robot(3));
      particles(i).landmarks(l).mu = [x; y];

      % get the Jacobian with respect to the landmark position
      [h, H] = measurement_model(particles(i), z(j));

      % Initialize the EKF covariance for this landmark
      % Using a large initial uncertainty
      particles(i).landmarks(l).sigma = 1000 * eye(2);

      % Indicate that this landmark has been observed
      particles(i).landmarks(l).observed = true;

    else

      % get the expected measurement
      [expectedZ, H] = measurement_model(particles(i), z(j));

      % Compute the measurement covariance
      S = H * particles(i).landmarks(l).sigma * H' + Q_t;

      % Calculate the Kalman gain
      K = particles(i).landmarks(l).sigma * H' / S;

      % Compute the error between the z and expectedZ (normalize the angle)
      innovation = [z(j).range - expectedZ(1); normalize_angle(z(j).bearing - expectedZ(2))];

      % Update the mean and covariance of the EKF for this landmark
      particles(i).landmarks(l).mu = particles(i).landmarks(l).mu + K * innovation;
      particles(i).landmarks(l).sigma = (eye(2) - K * H) * particles(i).landmarks(l).sigma;

      % Compute the likelihood of this observation
      % Using a Gaussian likelihood
      likelihood = exp(-0.5 * innovation' / S * innovation) / sqrt(2 * pi * det(S));
      particles(i).weight = particles(i).weight * likelihood;

    end

  end % measurement loop
end % particle loop

end
