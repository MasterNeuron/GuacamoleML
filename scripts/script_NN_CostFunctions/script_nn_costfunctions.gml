#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________________________________________

	Cost functions for MLP optimizers (Array, Grid). MLP Plus optimizer has them in extension, at the end of script are methods calling them.
	Calculates delta-error for output-layer, and returns average error.
	Remember that some cost-functions assume different ranges, like [0, 1] or [-1, 1]. 
	So you might need to use different activation functions for output-layer depent on cost-function you use.
	
	stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
	gombru.github.io/2018/05/23/cross_entropy_loss/
	towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
____________________________________________________________________________________________________________________________________
*/
#endregion


/*____________________________________________________________________________________________________________________________________
*/

/// @func	Delta(delta, predictions, targets);
/// @desc	Set output delta-error directly to given values. 
/// @param	{array}		delta			Error-delta Array to be updated.
/// @param	{array}		prediction		(Not actually used. Just to have same format as other cost-functions.)
/// @param	{array}		target			Array holding errors which will replace delta
function Delta(delta, prediction, targets) {
	var deltaCount = min(array_length(delta), array_length(targets));
		deltaCount = max(0, deltaCount);
	array_copy(delta, 0, targets, 0, deltaCount);
	return 0;
}

/*____________________________________________________________________________________________________________________________________
*/


/// @func	MeanSquare(delta, predictions, targets);
/// @desc	Mean Squared Error (MSE). 
/// @desc	Pretty General-purpose cost-function, but works better with regressional problems.
function MeanSquare(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
		error += sqr(prediction - target);
		delta[@i] = (prediction - target);
	}
    return .5 * error / iEnd;
}

/*____________________________________________________________________________________________________________________________________
*/	
	
/// @func	MeanAbsolute(delta, predictions, targets);
/// @desc	Mean absolute Error. 
function MeanAbsolute(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
		error += abs(prediction - target);
		delta[@i] = sign(prediction - target);
	}
    return error / iEnd;
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	MeanSquareLog(delta, predictions, targets);
/// @desc	Mean Squared Logarithmic Error. Variation of MSE.
function MeanSquareLog(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
		error += sqr(ln(prediction + 1) - ln(target + 1));
		delta[@i] = 2 * (ln(prediction + 1) - ln(target + 1)) / (prediction + 1);
	}
    return error / iEnd;
}
	
/*____________________________________________________________________________________________________________________________________
*/

/// @func	RootMeanSquare(delta, predictions, targets);
/// @desc	Root Mean Squared Error, (RMSE). -> sqrt(MSE)
/// @desc	Pretty General-purpose cost-function, but works better with regressional problems.
function RootMeanSquare(delta, predictions, target) {
	var error = sqrt(MeanSquare(delta, predictions, target));
	var iEnd = array_length(delta);
	var divider = 1 / (sqrt(iEnd) * error);
	for(var i = 0; i < iEnd; i++) {
		delta[@i] *= divider * .5;
	}
    return error;	
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	RootMeanSquareLog(delta, predictions, targets);
/// @desc	Root Mean Square Logarithmic Error.
function RootMeanSquareLog(delta, predictions, targets) {
	var error = sqrt(MeanSquareLog(delta, predictions, targets));
	var iEnd = array_length(delta);
	var divider = 1 / (sqrt(iEnd) * error);
	for(var i = 0; i < iEnd; i++) {
		delta[@i] *= divider * .5;
	}
    return error;	
}

/*____________________________________________________________________________________________________________________________________
*/	
	
/// @func	Huber(delta, predictions, targets, threshold);
/// @desc	Huber loss, cost function. This avoids problems (which MeanSquared would have) caused outliers in the data. Smooth Mean Absolute Error
/// @desc	en.wikipedia.org/wiki/Huber_loss
/// @param	{array}	delta
/// @param	{array}	predictions
/// @param	{array}	target
/// @param	{real}	threshold	When smaller: behaves like MeanSquared, when larger: behaves like MeanAbsolute.
function Huber(delta, predictions, targets, threshold) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];		
		if (abs(prediction - target) <= threshold) {
			error += .5 * sqr(prediction - target);
			delta[@i] = (prediction - target);
		} else {
			error += threshold * abs(prediction - target) - .5 * sqr(threshold);
			delta[@i] = threshold * (prediction - target) / abs(prediction - target);
		}
    }
    return error;
}	

/*____________________________________________________________________________________________________________________________________
*/
		
/// @func	Hinge(delta, predictions, targets);
/// @desc	Is used for "maximum-marging" classification.
/// @desc	https://en.wikipedia.org/wiki/Hinge_loss
function Hinge(delta, predictions, targets) {
	var error = 0;
	var value;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {	
        value = predictions[i] * targets[i];
        error += max(0, 1 - value);
        delta[@i] = (value < 1) ? -targets[i] : 0;	// Actually not differentiable at 1.
    }
    return error;
}
	
/*____________________________________________________________________________________________________________________________________
*/
	
/// @func	SmoothHinge(delta, predictions, targets);
/// @desc	Smoothed Hinge.
function SmoothHinge(delta, predictions, targets) {
	var error = 0;
	var prediction, target, value;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];	
        value = prediction * target;
		if (value <= 0) {
	        error += .5 - value;
	        delta[@i] = - target;
		} else if (value >= 1){
			error += 0;
			delta[@i] = 0;
		} else {
			error += .5 * sqr(1 - value);
			delta[@i] = - target * (1 - value);
		}
    }
    return error;
}
		
/*____________________________________________________________________________________________________________________________________
*/
	
/// @func	QuadHinge(delta, predictions, targets, parameter);
/// @desc	Quadratically Smoothed Hinge.
function  QuadHinge(delta, predictions, targets, hyperParameter) {
	var error = 0;
	var prediction, target, value;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];	
        value = prediction * target;
		if (value >= 1 - hyperParameter) {
	        error += (1 / (2*hyperParameter)) * sqr(max(0, 1 - value));
	        delta[@i] = (value >= 1) ? 0 : (target * (value - 1) / hyperParameter);
		} else if (value >= 1){
			error += 1 - (hyperParameter/2) - value;
			delta[@i] = (-1) * target;
		}
    }
    return error;
}

/*____________________________________________________________________________________________________________________________________
*/
	
/// @func	LogCosh(delta, predictions, targets);
/// @desc	Log-Cosh loss. 
function LogCosh(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];	
        error += ln( .5 * (exp(prediction - target) + exp(-(prediction - target))));	// ln( cosh(predicted - desired));
        delta[@i] = Tanh(prediction - target);
    }
    return error;
}
	
/*____________________________________________________________________________________________________________________________________
*/

/// @func	CrossEntropy(delta, predictions, targets);
/// @desc	Cross Entropy, using general formula. Predictions should be probability vector -> Array values should sum up total of 1.
/// @desc	 -> Use soft-max or similiar activation function for output-layer.
/// @desc	Categorial cross-entropy is simplified version, which can achieved with target array having correct answer 1 and others 0 (sums up to 1.). eg. array looks like [1, 0, 0, ...]
function CrossEntropy(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];	
        error += (-1) * target * ln(prediction);
        delta[@i] = (-1) * target / prediction;
    }
    return error;
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	CategorialCE(delta, predictions, targets);
/// @desc	Categorial Cross Entropy. Special case of cross-entropy.
/// @desc	Special as target-probability vector is 100% in one class.
/// @desc		-> Target should be one-hot vector (only one "1", others "0") -> eg. [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0];
/// @desc	Predictions should be probablity distribution -> sum up to 1. -> eg. [.7, .1, .2], [.0, .9, .1], [.3, .5, .2]
/// @desc	Because how predictions and targets should be defined, we can only care about "hot" prediction.
function CategorialCE(delta, predictions, targets) {
	var error = 0;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		delta[@i] = 0;
    }											// Normally: error += -target * ln(prediction)
	// Find index for hot						//		
	for(var i = 0; i < iEnd; i++) {				// But as targets array is type of [1, 0, 0, 0]
		if (targets[i] == 1) {					// Wrong: error += -0 * ln(prediction)		-> error += 0;
			error = (-1) * ln(predictions[i]);	// Correct: error += -1 * ln(prediction)	-> error += -ln(prediction)
			delta[@i] = (-1) / predictions[i];	// This can be simplified!
			break;								// Just search "hot" index.
		}
    }
    return error;
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	BinaryCE(delta, predictions, targets);
/// @desc	Binary Cross Entropy
/// @desc	Also Know as: Log-loss, Bernoulli negative log-likelihood,  or Sigmoid Cross-Entropy loss.
/// @desc	Specific case of cross-entropy. Target is either 0 or 1.
function BinaryCE(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
		if (target == 0) {
			error += (-1) * (1 - target) * ln(1 - prediction);
			delta[@i] = (-1) * (target - 1) / (1 - prediction);	
		} else {
			error += (-1) * target * ln(prediction);
			delta[@i] = (-1) * target / prediction;
		}
    }
    return error;
}
	
/*____________________________________________________________________________________________________________________________________
*/

/// @func	MultiCE(delta, predictions, targets);
/// @desc	Multi-Label Cross Entropy. Target can represent multiple classes at once.
/// @desc	Computes binary cross-entropy for each class separately and then sum them up.
/// @desc	This is used for multi-label categorization
/// @desc	Target and Prediction are NOT probability vector. Individual values range [0, 1] but don't need to sum up to 1.
function MultiCE(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];	
        error += (-1) * (target * ln(prediction) + (1 - target) * ln(1 - prediction));
        delta[@i] = (prediction - target) / ((1 - prediction) * prediction);
    }
    return error;
}
		
/*____________________________________________________________________________________________________________________________________
*/

/// @func	Exponential(delta, predictions, targets, parameter);
/// @desc	Need to have some hyperparameter. Usually play around until you find good one.
function Exponential(delta, predictions, targets, hyperParameter) {
	var error = 0;
	var iEnd = array_length(delta);
	var divider = 1 / hyperParameter;
	// Calculate error
	for(var i = 0; i < iEnd; i++) {
		error += sqr(predictions[i] - targets[i]);
	}
	error = hyperParameter * exp(error * divider);

	// Calculate delta. Use derivative of Exp.Cost.
	for(var i = 0; i < iEnd; i++) {
		delta[@i] = (2 * divider) * (predictions[i] - targets[i]) * error;	// Have to calculate error first 
	}
	return error;
}
	
/*____________________________________________________________________________________________________________________________________
*/

/// @func	Hellinger(delta, predictions, targets);
/// @desc	Hellinger distance. Needs positive value between [0,1] 
function Hellinger(delta, predictions, targets) {  
	var error = 0;
	var twoSqrt = sqrt(2);	// Cache
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = sqrt(predictions[i]);
		target = sqrt(targets[i]);
        error += sqr(prediction - target);
        delta[@i] = (prediction - target) / (twoSqrt * prediction); 
    }
    return (error / twoSqrt);
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	KullbackLeibler(delta, predictions, targets);
/// @desc	Kullback-Leibler divergence.
/// @desc	Aka. Information Divergence, Information Gain, Relative entropy, KLIC divergence or KL Divergence
function KullbackLeibler(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
        error += target * ln(target / prediction);
        delta[@i] = (-1) * (target / prediction); 
    }
    return error;
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	GeneralizedKL(delta, predictions, targets);
/// @desc	Generalized Kullback-Leibler divergence.
/// @desc	www.machinecurve.com/index.php/2019/12/21/how-to-use-kullback-leibler-divergence-kl-divergence-with-keras/
function GeneralizedKL(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var predictionSum = 0;
	var targetSum = 0;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
        predictionSum += prediction;
        targetSum += target;
        error += target * ln(target / prediction);
        delta[@i] = (prediction - target) / prediction;
    }
    return error - targetSum + predictionSum;	
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	ItakuraSaito(delta, predictions, targets);
/// @desc	Itakura-Saito Distance.
function ItakuraSaito(delta, predictions, targets) {
	var error = 0;
	var prediction, target;
	var iEnd = array_length(delta);
	for(var i = 0; i < iEnd; i++) {
		prediction = predictions[i];
		target = targets[i];
        error += (target / prediction) - ln(target / prediction) - 1;
        delta[@i] = (prediction - target) / sqr(prediction);
    }
    return error;
}

/*____________________________________________________________________________________________________________________________________
*/








