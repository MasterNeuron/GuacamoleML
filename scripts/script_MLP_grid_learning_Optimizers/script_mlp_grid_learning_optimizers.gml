#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	GRADIENT DESCENT OPTIMIZERS
	 
	Use these with mlp methods like : mlp.Optimizer(Stochastic);
	
	Normal Gradient Descent calculates average gradient of all example, and then update weights.
	-> This is computationally heavy and slow. There are several optimizers for this.
	-> For example take mini-batches: calculate gradient from 1 or more examples.
	Minibatching works for every optimizer included here
	-> Backpropagate as many examples you want, then use Apply.

	(Weights and Biases need to be updated with accessors "bias[@i][@j]" so we directly modify target MLP.)
	(Others are in accessors as GML had some bug, which origin I don't, so I put accessors in everything to make sure it doesn't happen.)
		(-> Bug was clear when training with gradient descent, and networks didn't just learn without using accessors in "right" places, which should not have needed.)
	(start from i=1, because input has no gradients and input-biases are not used.).
	(You could make these optimizers work similiarly to Array with grid accessor like: [# j, k])
	
	https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
____________________________________________________________________________________________________
*/
#endregion

/// @func	___optimizer_grid(mlp);
function ___optimizer_grid(mlp) constructor {
	___gradients_grid_ini(mlp);
	
	/// @func	Backward(); 
	/// @desc	Backpropagation. Updates delta-errors, moving through from output towards input layer.
	static Backward = function() {
		___gradients_grid_backpropagation();
		return delta[0];
	}
	
	/// @func	Cost(costFunction, target, ?parameter);
	/// @desc	Calculates error for current output, returns total error. Updates delta-error for output-layer.
	/// @param	{function}	costFunction	
	/// @param	{array}		target			MLP values are compared against given target values
	/// @param	{real}		?parameter		Optional hyperparameter for cost-function.
	static Cost = function(costFunction, target, hyperParameter) {
		hyperParameter = is_undefined(hyperParameter) ? .5 : hyperParameter;
		var last = layerCount-1;
		return costFunction(delta[last], output[last], target, hyperParameter);
	}
	
	/// @func	Delta(target);
	/// @desc	Sets output-delta error to given delta-array. Used for linking MLP's together.
	/// @param	{array}		target
	static Delta = function(deltaTarget) {
		return Delta(delta[layerCount-1], NULL, deltaTarget);
	}
	
	/// @func	Decay(rate);
	/// @desc	Simple weight decaying function. Use this after "Apply()"
	/// @param	{real}	rate	How fast decays. For example ".01".
	static Decay = function(rate) {
		___optimizer_grid_weight_decay(rate);
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_stochastic(mlp);
/// @desc	Stochastic Gradient Descent -optimizer. 
/// @desc	Basic optimizer: Takes minibatch of examples and calculates gradients from them. Divided with 'trainingSession' we get average gradient of batch.
/// @param	{mlp_grid}	mlp
function ___optimizer_grid_stochastic(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.STOCHASTIC;
	
	/// @func	Destroy();
	static Destroy = function() {
		___gradients_grid_destroy();
	}
	
	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, jEnd, kEnd, J, K;
		learnRate = learnRate / trainingSession;	// To get average of several trainings (otherwise gradients are just sum of them).
		trainingSession = 0;						// Counting starts again.
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			// Update weights
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			ds_grid_multiply_region(gradients[i], 0, 0, J, K, -learnRate);
			ds_grid_add_grid_region(weights[i], gradients[i], 0, 0, J, K, 0, 0);
			ds_grid_clear(gradients[i], 0);
			// Update biases
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j];
				deltaSum[@i][@j] = 0;
			}
		}
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_momentum(mlp, momentumRate);
/// @desc	Momentum Gradient descent. Similiar to Stochastic, but has gradients have momentum.
/// @param	{mlp_grid}	mlp
/// @param	{real}		momentumRate	Value between 0-1. Think about rolling ball, this keeps learning 'momentum'.
function ___optimizer_grid_momentum(mlp, argMomentumRate) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.MOMENTUM;

	// Initialize momentums for weights
	momentumRate = argMomentumRate;
	momentums = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		momentums[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(momentums[i]);
		}
		momentums = NULL;
		___gradients_grid_destroy();
	}	
	
	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, jEnd, kEnd, J, K;
		learnRate = learnRate / trainingSession;
		trainingSession = 0;
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			// Update weights
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			ds_grid_set_grid_region(calculator, momentums[i],	0, 0, J, K, 0, 0);				//
			ds_grid_multiply_region(calculator,					0, 0, J, K, momentumRate);		//
			ds_grid_add_grid_region(calculator, gradients[i],	0, 0, J, K, 0, 0);				//
			ds_grid_set_grid_region(momentums[i], calculator,	0, 0, J, K, 0, 0);				// momentums = momentums * rate + gradients
			ds_grid_multiply_region(calculator,					0, 0, J, K, -learnRate);		//
			ds_grid_add_grid_region(weights[i], calculator,		0, 0, J, K, 0, 0);				// weights += momentums * (-learnRate)
			ds_grid_clear(gradients[i], 0);														// Clear for next round. Momentum stays.
			
			for(j = 0; j < layerSizes[i]; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j];
				deltaSum[@i][@j] = 0;
			}
		}
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_nesterov(mlp, momentumRate);
/// @desc	Nesterov Accelerated Gradient Descent. Similiar to momentum, but takes momentum at different time.
/// @param	{mlp_grid}	mlp
/// @param	{real}		momentumRate	Value between 0-1. Think about rolling ball, this keeps learning 'momentum'. 
function ___optimizer_grid_nesterov(mlp, argMomentumRate) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.NESTEROV;

	// Initialize momentums for weights
	momentumRate = argMomentumRate;
	momentums = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		momentums[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(momentums[i]);
		}
		momentums = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, jEnd, kEnd, J, K;
		learnRate = learnRate / trainingSession;
		trainingSession = 0;
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			// Update momentums.
			ds_grid_set_grid_region(calculator, momentums[i],	0, 0, J, K, 0, 0);				// old momentum = momentum							Save this for later, for updating weights
			ds_grid_multiply_region(momentums[i],				0, 0, J, K, momentumRate);		// new momentum = momentum * momentumRate			Update 
			ds_grid_multiply_region(gradients[i],				0, 0, J, K, -learnRate);		// set gradient = gradient * learnRate * (-1)		Update (-1 to go down the gradient)
			ds_grid_add_grid_region(momentums[i], gradients[i],	0, 0, J, K, 0, 0);				// new momentum = gradient + momentum				Calculate new momentum.
			// Update weights																
			ds_grid_multiply_region(calculator,					0, 0, J, K, -1);				// Need to (momentum - oldMomentum), but only have ds_grid_add, so we multiply with -1 first instead
			ds_grid_add_grid_region(calculator, momentums[i],	0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,					0, 0, J, K, momentumRate);
			ds_grid_add_grid_region(calculator, momentums[i],	0, 0, J, K, 0, 0);
			ds_grid_add_grid_region(weights[i], calculator,		0, 0, J, K, 0, 0);
			ds_grid_clear(gradients[i], 0);	// Clear for next round. Momentum stays.

			// Update biases.
			for(j = 0; j < layerSizes[i]; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j];
				deltaSum[@i][@j] = 0;
			}
		}
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_adam(mlp);
/// @desc	Adaptive Moment Estimation. 'Adam' is optimization for Gradient descent, which combines combines Momentum and AdaGrad.
/// @param	{mlp_grid}	mlp
function ___optimizer_grid_adam(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.ADAM;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		twoMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}

	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
			ds_grid_destroy(twoMoment[i]);
		}	
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, two, grad;
		var epsilon = .001;										// To avoid dividing by 0.
		var divider = 1 / trainingSession;						// Caching division.
		var oneDivider = 1 / (1 - power(betaOne, iteration));	// 
		var twoDivider = 1 / (1 - power(betaTwo, iteration));	// 

		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one	= oneMoment[i];
			two = twoMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}			
			
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);	
			// One moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaOne);
			ds_grid_multiply_region(one,					0, 0, J, K, betaOne);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			// Two moment.
			ds_grid_set_grid_region(calculator, grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaTwo);
			ds_grid_multiply_region(two,					0, 0, J, K, betaTwo);
			ds_grid_add_grid_region(two, calculator,		0, 0, J, K, 0, 0);

			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				calculator[# j, k] = (1 / (sqrt(two[# j, k] * twoDivider) + epsilon));
			}}
			
			// Update weights
			ds_grid_multiply_region(calculator,				0, 0, J, K, -learnRate * oneDivider);
			ds_grid_multiply_grid_region(calculator, one,	0, 0, J, K, 0, 0);
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);
			
			// Clear gradients
			ds_grid_clear(grad, 0);
		}
				
		// Reset gradients for next training session. Keep momentums
		trainingSession = 0;	// Counting starts again.
		iteration++;			// Done one Adam-iteration more
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_adagrad(mlp);
/// @desc	Adaptive gradient algorithm, AdaGrad. 
/// @param	{mlp_array}	mlp
function ___optimizer_grid_adagrad(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.ADAGRAD;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);	// Modifies learning rate depent on previous values of gradients deltas
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
		}	
		oneMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, grad;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
						
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one = oneMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}		

			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);	
			
			// Moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// sqr(grad)
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			
			// Calculate
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, -learnRate);

			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				calculator[# j, k] /= sqrt(one[# j, k]) + epsilon;
			}}
			
			// Update weights
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);

			// Clear gradients
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_adadelta(mlp, parameter);
/// @desc	"Adaptive delta". Extension of AdaGrad. Similiar to RMSprop. 
/// @desc	AdaDelta removes use of learning parameter by replacing it with "Exponential moving average of squared deltas". 
/// @param	{mlp_array}	mlp
/// @param	{real}		parameter	Decaying rate, usually around .9
function ___optimizer_grid_adadelta(mlp, parameter) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.ADADELTA;
	decaying_rate = parameter;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		twoMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
			ds_grid_destroy(twoMoment[i]);
		}	
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, two, grad;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one	= oneMoment[i];
			two = twoMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}			
			
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);	
			
			// One moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, (1-decaying_rate));
			ds_grid_multiply_region(one,					0, 0, J, K, decaying_rate);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			
			// Update weights
			ds_grid_set_grid_region(calculator, weights[i],	0, 0, J, K, 0, 0);	// Store old weights temporaly
			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				weights[i][# j, k] += - grad[# j, k] * sqrt(two[# j, k] + epsilon) / (sqrt(one[# j, k] + epsilon));
			}}
			
			// Two moment.	(update after weights).
			ds_grid_multiply_region(calculator,				0, 0, J, K, -1);
			ds_grid_add_grid_region(calculator, weights[i],	0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, calculator,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1 - decaying_rate);
			ds_grid_multiply_region(two,					0, 0, J, K, decaying_rate);
			ds_grid_add_grid_region(two, calculator,		0, 0, J, K, 0, 0);
			
			// Clear gradients
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_adamax(mlp);
/// @desc	Adaptation of Adam optimizer. Proposed default learning rate is .002
/// @param	{mlp_array}	mlp
function ___optimizer_grid_adamax(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.ADAMAX;
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);	// exponential moving average of gradients
	twoMoment = array_create(layerCount, NULL);	// exponential moving average of past p-norm of gradients.
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		twoMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
			ds_grid_destroy(twoMoment[i]);
		}	
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, two, grad;
		var divider = 1 / trainingSession;						// Cache division
		var oneDivider = 1 / (1 - power(betaOne, iteration));	//
				
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one	= oneMoment[i];
			two = twoMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}			
			
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);
			
			// One moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaOne);
			ds_grid_multiply_region(one,					0, 0, J, K, betaOne);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			
			// Two moment.
			ds_grid_multiply_region(two,					0, 0, J, K, betaTwo);
			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				two[# j, k] = max(two[# j, k], abs(grad[# j, k]));
			}}
			
			// Calculate
			//		-(learnRate / twoMoment[i][# j, k]) * (oneMoment[i][# j, k] * oneDivider)
			ds_grid_set_grid_region(calculator, one,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, -learnRate * oneDivider);	
			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				calculator[# j, k] /= two[# j, k];
			}}			
			
			// Update weights
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);			
			
			// Clear gradients
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
		iteration++;
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_nadam(mlp);
/// @desc	Nesterov Adaptive Moment Estimation. Combines Nesterov and Adam optimizers.
/// @param	{mlp_array}	mlp
function ___optimizer_grid_nadam(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.NADAM;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		twoMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
			ds_grid_destroy(twoMoment[i]);
		}	
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, two, grad;
		var epsilon = .001;	// To avoid dividing by 0.
		var divider = 1 / trainingSession;						// Caching.
		var oneDivider = 1 / (1 - power(betaOne, iteration));	//
		var twoDivider = 1 / (1 - power(betaTwo, iteration));	//
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one	= oneMoment[i];
			two = twoMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}	
				
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);
			
			// One moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaOne);
			ds_grid_multiply_region(one,					0, 0, J, K, betaOne);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			
			// Two moment.
			ds_grid_set_grid_region(calculator, grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaTwo);
			ds_grid_multiply_region(two,					0, 0, J, K, betaTwo);
			ds_grid_add_grid_region(two, calculator,		0, 0, J, K, 0, 0);

			// Calculate 
			// Next lines do following calculation : 
			//		-learnRate / (sqrt(twoMoment[i][# j, k] * twoDivider) + epsilon)) * (betaOne * oneMoment[i][# j, k] * oneDivider + ((1 - betaOne) * oneDivider) * gradient);
			for(j = 0; j < jEnd; j++) {	// Can't do with regions :(
			for(k = 0; k < kEnd; k++) {
				calculator[# j, k] = -learnRate / (sqrt(two[# j, k] * twoDivider) + epsilon);
			}}
			
			ds_grid_multiply_region(grad,					0, 0, J, K, (1 - betaOne) * oneDivider);
			ds_grid_multiply_grid_region(grad, calculator,	0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, betaOne * oneDivider);
			ds_grid_multiply_grid_region(calculator, one,	0, 0, J, K, 0, 0);
			ds_grid_add_grid_region(calculator, grad,		0, 0, J, K, 0, 0);
			
			// Update weights
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);				
			
			// Clear gradients
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
		iteration++;			// Done one Adam-iteration more
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_rmsprop(mlp, parameter);
/// @desc	Extension of AdaGrad, similiar to AdaDelta.
/// @param	{mlp_array}	mlp
/// @param	{real}		parameter	Usually around .9, decaying rate.
function ___optimizer_grid_rmsprop(mlp, parameter) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.RMSPROP;
	decaying_rate = parameter;
	
	// Initialize
	momentums = array_create(layerCount, NULL);
	for(var i = 1; i < layerCount; i++) {
		momentums[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(momentums[i]);
		}	
		momentums = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var grad, one;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {			
			one = momentums[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}					
				
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);	
			
			// Moment.
			ds_grid_set_grid_region(calculator, grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-decaying_rate);
			ds_grid_multiply_region(one,					0, 0, J, K, decaying_rate);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);

			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				calculator[# j, k] = -learnRate / (sqrt(one[# j, k] + epsilon));
			}}
			
			// Update weights
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);
			
			// Clear gradients		
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_amsgrad(mlp);
/// @desc	Variant of Adam.
/// @param	{mlp_array}	mlp
function ___optimizer_grid_amsgrad(mlp) : ___optimizer_grid(mlp) constructor {
	type = OptimizerType.AMSGRAD;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	maxMoment = array_create(layerCount, NULL);	// Stores largest last twoMoment -> Always grows.
	for(var i = 1; i < layerCount; i++) {
		oneMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		twoMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
		maxMoment[@i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
	}
	
	/// @func	Destroy();
	static Destroy = function() {
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(oneMoment[i]);
			ds_grid_destroy(twoMoment[i]);			
			ds_grid_destroy(maxMoment[i]);
		}	
		oneMoment = NULL;
		twoMoment = NULL;
		maxMoment = NULL;
		___gradients_grid_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd, J, K;
		var one, two, mxm, grad;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Caching division.
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			one	= oneMoment[i];
			two = twoMoment[i];
			mxm = maxMoment[i];
			grad = gradients[i];
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			
			// Update bias
			for(j = 0; j < jEnd; j++) {
				bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
				deltaSum[@i][@j] = 0;
			}			
			
			// Get mean gradients.
			ds_grid_multiply_region(grad,					0, 0, J, K, divider);			
	
			// One moment.									
			ds_grid_set_grid_region(calculator,	grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaOne);
			ds_grid_multiply_region(one,					0, 0, J, K, betaOne);
			ds_grid_add_grid_region(one, calculator,		0, 0, J, K, 0, 0);
			
			// Two moment.
			ds_grid_set_grid_region(calculator, grad,		0, 0, J, K, 0, 0);
			ds_grid_multiply_grid_region(calculator, grad,	0, 0, J, K, 0, 0);	// Make it squared. 
			ds_grid_multiply_region(calculator,				0, 0, J, K, 1-betaTwo);
			ds_grid_multiply_region(two,					0, 0, J, K, betaTwo);
			ds_grid_add_grid_region(two, calculator,		0, 0, J, K, 0, 0);

			// Max moment. Also update calculator.
			for(j = 0; j < jEnd; j++) {
			for(k = 0; k < kEnd; k++) {
				mxm[# j, k] = max(mxm[# j, k], two[# j, k]);
				calculator[# j, k] = -learnRate / (sqrt(mxm[# j, k]) + epsilon);	
			}}
			ds_grid_multiply_grid_region(calculator, one,	0, 0, J, K, 0, 0);

			// Update weights
			ds_grid_add_grid_region(weights[i], calculator,	0, 0, J, K, 0, 0);
			
			// Clear gradients	
			ds_grid_clear(grad, 0);
		}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_grid_weight_decay(decayAmount);
/// @desc	Regularization method for avoiding over-fitting.
function ___optimizer_grid_weight_decay(decayAmount) {
	var i, j, k, J, K;

	// Update weights and biases.
	for(i = 1; i < layerCount; i++) {
		J = layerSizes[i];
		K = layerSizes[i-1];
		ds_grid_multiply_region(weights[i], 0, 0, J, K, (1-decayAmount));
	}
}

/*____________________________________________________________________________________________________
*/













