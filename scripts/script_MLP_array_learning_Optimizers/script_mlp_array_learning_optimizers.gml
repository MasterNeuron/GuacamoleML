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
	
	https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
	https://ruder.io/optimizing-gradient-descent/index.html
____________________________________________________________________________________________________
*/
#endregion

/// @func	___optimizer_array(mlp);
function ___optimizer_array(mlp) constructor {
	___gradients_array_ini(mlp);
	
	/// @func	Backward(); 	
	/// @desc	Backpropagation. Updates delta-errors, moving through from output towards input layer.
	static Backward = function() {
		___gradients_array_backpropagation();
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
		return Delta(delta[layerCount-1], deltaTarget, deltaTarget);
	}
	
	/// @func	Decay(rate);
	/// @desc	Simple weight decaying function. Use this after "Apply()"
	/// @param	{real}	rate	How fast decays. For example ".01".
	static Decay = function(rate) {
		___optimizer_array_weight_decay(rate);
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_stochastic(mlp);
/// @desc	Stochastic Gradient Descent -optimizer. 
/// @desc	Basic optimizer: Takes minibatch of examples and calculates gradients from them. Divided with 'trainingSession' we get average gradient of batch.
/// @param	{mlp_array}	mlp
function ___optimizer_array_stochastic(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.STOCHASTIC;
	
	/// @func	Destroy();
	static Destroy = function() {
		___gradients_array_destroy();
	}
	
	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		learnRate = -learnRate / trainingSession;	// To get average of several trainings (otherwise gradients are just sum of them).
		trainingSession = 0;						// Counting starts again.
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += learnRate * deltaSum[i][j];
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			weights[@i][@j][@k] += learnRate * gradients[i][j][k];
			gradients[@i][@j][@k] = 0;
		}}}
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_momentum(mlp, momentumRate);
/// @desc	Momentum Gradient descent. Similiar to Stochastic, but has gradients have momentum.
/// @param	{mlp_array}	mlp
/// @param	{real}		momentumRate	Value between 0-1. Think about rolling ball, this keeps learning 'momentum'.
function ___optimizer_array_momentum(mlp, argMomentumRate) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.MOMENTUM;

	// Initialize momentums for weights
	momentumRate = argMomentumRate;
	momentums = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		momentums[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		momentums[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		momentums = NULL;
		___gradients_array_destroy();
	}	
	
	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		learnRate = -learnRate / trainingSession;
		trainingSession = 0;
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += learnRate * deltaSum[i][j];
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			momentums[@i][@j][@k] = momentums[i][j][k] * momentumRate + gradients[i][j][k];	// Momentum is updated by gradient and previous momentum.
			weights[@i][@j][@k] += momentums[i][j][k] * learnRate;		// weight is updated according to momentum
			gradients[@i][@j][@k] = 0;
		}}}
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_nesterov(mlp, momentumRate);
/// @desc	Nesterov Accelerated Gradient Descent. Similiar to momentum, but takes momentum at different time.
/// @param	{mlp_array}	mlp
/// @param	{real}		momentumRate	Value between 0-1. Think about rolling ball, this keeps learning 'momentum'. 
function ___optimizer_array_nesterov(mlp, argMomentumRate) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.NESTEROV;

	// Initialize momentums for weights
	momentumRate = argMomentumRate;
	momentums = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		momentums[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		momentums[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		momentums = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var oldMomentum, newMomentum;
		learnRate = -learnRate / trainingSession;
		trainingSession = 0;
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += learnRate * deltaSum[i][j];
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			oldMomentum = momentums[i][j][k];
			momentums[i][j][k] = learnRate * gradients[i][j][k] + oldMomentum * momentumRate;
			newMomentum = momentums[i][j][k];
			weights[@i][@j][@k] += newMomentum + momentumRate * (newMomentum - oldMomentum);
			gradients[@i][@j][@k] = 0;
		}}}
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_adam(mlp);
/// @desc	Adaptive Moment Estimation. 'Adam' is optimization for Gradient descent, which combines combines Momentum and AdaGrad.
/// @param	{mlp_array}	mlp
function ___optimizer_array_adam(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.ADAM;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient, unbiasOne, unbiasTwo;
		var epsilon = .001;	// To avoid dividing by 0.
		var divider = 1 / trainingSession;						// Caching.
		var oneDivider = 1 / (1 - power(betaOne, iteration));	// 
		var twoDivider = 1 / (1 - power(betaTwo, iteration));	// 
		
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			// Calculate momentum
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;
			oneMoment[@i][@j][@k] = betaOne * oneMoment[i][j][k] + (1 - betaOne) * gradient;
			twoMoment[@i][@j][@k] = betaTwo * twoMoment[i][j][k] + (1 - betaTwo) * gradient * gradient;
			// Bias correction
			unbiasOne = oneMoment[i][j][k] * oneDivider;
			unbiasTwo = twoMoment[i][j][k] * twoDivider;
			// AdaGrad.
			weights[@i][@j][@k] += -learnRate * unbiasOne / (sqrt(unbiasTwo) + epsilon);
		}}}
				
		trainingSession = 0;	// Counting starts again.
		iteration++;		// Done one Adam-iteration more
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_adagrad(mlp);
/// @desc	Adaptive gradient algorithm, AdaGrad. 
/// @param	{mlp_array}	mlp
function ___optimizer_array_adagrad(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.ADAGRAD;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);	// Modifies learning rate depent on previous values of gradients deltas
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;	
			
			oneMoment[@i][@j][@k] += sqr(gradient);
			weights[@i][@j][@k] += - learnRate * gradient / (sqrt(oneMoment[i][j][k] + epsilon));
		}}}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_adadelta(mlp, parameter);
/// @desc	"Adaptive delta". Extension of AdaGrad. Similiar to RMSprop. 
/// @desc	AdaDelta removes use of learning parameter by replacing it with "Exponential moving average of squared deltas". 
/// @param	{mlp_array}	mlp
/// @param	{real}		parameter	Decaying rate, usually around .9
function ___optimizer_array_adadelta(mlp, parameter) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.ADADELTA;
	decaying_rate = parameter;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient, oldWeight, oneCalc, twoCalc;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[i][j] = 0;
		for(k = 0; k < kEnd; k++) {
			gradient = gradients[i][j][k] * divider;
			gradients[i][j][k] = 0;	

			oneCalc = oneMoment[@i][@j][@k] * decaying_rate + (1 - decaying_rate) * sqr(gradient);
			twoCalc = twoMoment[@i][@j][@k];	// Use old for weights, update after it.
			oldWeight = weights[@i][@j][@k];

			weights[@i][@j][@k] += - gradient * sqrt(twoCalc + epsilon) / (sqrt(oneCalc + epsilon));
			
			// Update parameters.
			twoCalc = twoCalc * decaying_rate + (1 - decaying_rate) * sqr(weights[i][j][k] - oldWeight);
			oneMoment[@i][@j][@k] = oneCalc;
			twoMoment[@i][@j][@k] = twoCalc;
		}}}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_adamax(mlp);
/// @desc	Adaptation of Adam optimizer. Proposed default learning rate is .002
/// @param	{mlp_array}	mlp
function ___optimizer_array_adamax(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.ADAMAX;
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	
	// Initialize
	oneMoment = array_create(layerCount, NULL);	// exponential moving average of gradients
	twoMoment = array_create(layerCount, NULL);	// exponential moving average of past p-norm of gradients.
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient;
		var divider = 1 / trainingSession;						// Cache division
		var oneDivider = 1 / (1 - power(betaOne, iteration));	//
				
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[i][j] = 0;
		for(k = 0; k < kEnd; k++) {
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;
			oneMoment[@i][@j][@k] = oneMoment[i][j][k] * betaOne + (1 - betaOne) * gradient;
			twoMoment[@i][@j][@k] = max(betaTwo * twoMoment[i][j][k], abs(gradient));
			
			weights[@i][@j][@k] += - (learnRate / twoMoment[i][j][k]) * (oneMoment[i][j][k] * oneDivider);	
		}}}
				
		trainingSession = 0;	// Counting starts again.
		iteration++;
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_nadam(mlp);
/// @desc	Nesterov Adaptive Moment Estimation. Combines Nesterov and Adam optimizers.
/// @param	{mlp_array}	mlp
function ___optimizer_array_nadam(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.NADAM;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	iteration = 1;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		twoMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient, unbiasOne, unbiasTwo;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;						// Caching.
		var oneDivider = 1 / (1 - power(betaOne, iteration));	//
		var twoDivider = 1 / (1 - power(betaTwo, iteration));	//
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			// Calculate momentum
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;
			oneMoment[@i][@j][@k] = betaOne * oneMoment[i][j][k] + (1 - betaOne) * gradient;
			twoMoment[@i][@j][@k] = betaTwo * twoMoment[i][j][k] + (1 - betaTwo) * gradient * gradient;
			// Bias correction
			unbiasOne = oneMoment[i][j][k] * oneDivider;
			unbiasTwo = twoMoment[i][j][k] * twoDivider;
			// Update weights
			weights[@i][@j][@k] += - (learnRate / (sqrt(unbiasTwo) + epsilon)) * (betaOne * unbiasOne + ((1 - betaOne) * oneDivider) * gradient);
		}}}
				
		trainingSession = 0;	// Counting starts again.
		iteration++;			// Done one Adam-iteration more
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_rmsprop(mlp, parameter);
/// @desc	Extension of AdaGrad, similiar to AdaDelta.
/// @param	{mlp_array}	mlp
/// @param	{real}		parameter	Usually around .9, decaying rate.
function ___optimizer_array_rmsprop(mlp, parameter) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.RMSPROP;
	decaying_rate = parameter;
	
	// Initialize
	momentums = array_create(layerCount, NULL);
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		momentums[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		momentums[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		momentums = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient, momentum;
		var epsilon = .001;					// To avoid dividing by 0.
		var divider = 1 / trainingSession;	// Cache division
			
		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			gradient = gradients[i][j][k] * divider;
			momentum = momentums[i][j][k] * decaying_rate + (1 - decaying_rate) * sqr(gradient);
			gradients[@i][@j][@k] = 0;
			momentums[@i][@j][@k] = momentum;	// Update
			weights[@i][@j][@k] += - learnRate * gradient / (sqrt(momentum + epsilon));
		}}}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_amsgrad(mlp);
/// @desc	Variant of Adam.
/// @param	{mlp_array}	mlp
function ___optimizer_array_amsgrad(mlp) : ___optimizer_array(mlp) constructor {
	type = OptimizerType.AMSGRAD;
	
	// Initialize momentums for weights
	betaOne = .9;
	betaTwo = .999;
	oneMoment = array_create(layerCount, NULL);
	twoMoment = array_create(layerCount, NULL);
	maxMoment = array_create(layerCount, NULL);	// Stores largest last twoMoment -> Always grows.
	var i, j, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		oneMoment[@i] = array_create(jEnd, NULL);
		twoMoment[@i] = array_create(jEnd, NULL);
		maxMoment[@i] = array_create(jEnd, NULL);
	for(j = 0; j < jEnd; j++) {
		oneMoment[@i][@j] = array_create(kEnd, 0);
		twoMoment[@i][@j] = array_create(kEnd, 0);
		maxMoment[@i][@j] = array_create(kEnd, 0);
	}}
	
	/// @func	Destroy();
	static Destroy = function() {
		oneMoment = NULL;
		twoMoment = NULL;
		maxMoment = NULL;
		___gradients_array_destroy();
	}	

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		var i, j, k, jEnd, kEnd;
		var gradient;
		var epsilon = .001;	// To avoid dividing by 0.
		var divider = 1 / trainingSession; // Caching.

		// Update weights and biases.
		for(i = 1; i < layerCount; i++) {
			jEnd = layerSizes[i];
			kEnd = layerSizes[i-1];
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] += -learnRate * deltaSum[i][j] * divider;
			deltaSum[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			// Calculate momentum
			gradient = gradients[i][j][k] * divider;
			gradients[@i][@j][@k] = 0;
			oneMoment[@i][@j][@k] = betaOne * oneMoment[i][j][k] + (1 - betaOne) * gradient;
			twoMoment[@i][@j][@k] = betaTwo * twoMoment[i][j][k] + (1 - betaTwo) * gradient * gradient;
			maxMoment[@i][@j][@k] = max(maxMoment[i][j][k], twoMoment[i][j][k]);	// Only grows.
			
			// Update
			weights[@i][@j][@k] += -learnRate * oneMoment[i][j][k] / (sqrt(maxMoment[i][j][k]) + epsilon);
		}}}
				
		trainingSession = 0;	// Counting starts again.
	}	
}

/*____________________________________________________________________________________________________
*/

/// @func	___optimizer_array_weight_decay(decayAmount);
/// @desc	Regularization method for avoiding over-fitting.
function ___optimizer_array_weight_decay(decayAmount) {
	var i, j, k, jEnd, kEnd;

	// Update weights and biases.
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
	for(k = 0; k < kEnd; k++) {
		weights[@i][@j][@k] *= (1 - decayAmount);
	}}}
}

/*____________________________________________________________________________________________________
*/






















