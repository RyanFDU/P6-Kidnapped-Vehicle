/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

// import namespaces
using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

// Use a random number generator
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  // Check if it has initialized already
  if (is_initialized) {
    return;
  }

  // Create Gaussian distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100;  // TODO: Set the number of particles

  // Generate particles with Gaussian distribution on GPS value
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;   //Using index as particle id
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    // Store the particles and weights
    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Create a Gaussian distribution for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Predict each particle after timestep delta_t
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) >= 0.00001) {
      particles[i].x += 
        (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y +=
        (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // Add noise here
    particles[i].x     += dist_x(gen);
    particles[i].y     += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  int n_ob = observations.size();
  int n_pr = predicted.size();
  
  for (int i = 0; i < n_ob; ++i) {
    double min_dist = std::numeric_limits<double>::max();
    int map_id = -1;

    // Complexity is o(ij);
    for (int j = 0; j < n_pr; ++j) {
      // Calculate the distance
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // Update the result
      if (distance < min_dist) {
          min_dist = distance;
          map_id = predicted[j].id;
      }
    }
    
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   vector<LandmarkObs> &observations, 
                                   Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double weight_normalizer = 0.0;

  for (int i = 0; i < num_particles; ++i) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // Transform observations to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs transformed_obs;
      transformed_obs.id = j;
      transformed_obs.x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
      transformed_obs.y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
      transformed_observations.push_back(transformed_obs);
    }

    // Filter landmarks only within the current sensor range
    vector<LandmarkObs> predicted_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((p_x - current_landmark.x_f)) <= sensor_range) &&
          (fabs((p_y - current_landmark.y_f)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }

    // Associate observations to predicted landmarks using nearest neighbor algorithm
    dataAssociation(predicted_landmarks, transformed_observations);

    // Calculate the weight of each particle using Multivariate Gaussian distribution
    particles[i].weight = 1.0;  // Reset the weights to 1.0

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double sig_x_2 = pow(sig_x, 2);
    double sig_y_2 = pow(sig_y, 2);
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    for (int k = 0; k < transformed_observations.size(); ++k) {
      double trans_x = transformed_observations[k].x;
      double trans_y = transformed_observations[k].y;
      double trans_id = transformed_observations[k].id;
      double multi_prob = 1.0;

      for (int l = 0; l < predicted_landmarks.size(); ++l) {
        double pred_x = predicted_landmarks[l].x;
        double pred_y = predicted_landmarks[l].y;
        double pred_id = predicted_landmarks[l].id;

        if (pred_id == trans_id) {
          multi_prob = gauss_norm * exp(-1.0 * ((pow((trans_x - pred_x), 2)/(2.0 * sig_x_2)) + (pow((trans_y - pred_y), 2)/(2.0 * sig_y_2))));
          particles[i].weight *= multi_prob;
          break;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }

  // Normalize the weights of all particles
  for (int i = 0; i < particles.size(); ++i) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Using the Resampling Wheel given by Sebastian
  vector<Particle> resampled_particles;

	uniform_int_distribution<int> particle_index(0, num_particles - 1);
	int index = particle_index(gen);
	
	double beta = 0.0;
	
	double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());
	
	for (int i = 0; i < particles.size(); ++i) {
		uniform_real_distribution<double> random_weight(0.0, max_weight_2);
		beta += random_weight(gen);

	  while (beta > weights[index]) {
	    beta -= weights[index];
	    index = (index + 1) % num_particles;
	  }
	  resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}