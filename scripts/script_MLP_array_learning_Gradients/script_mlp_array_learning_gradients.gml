#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Creating gradients-structure from mlp. These don't need to be in mlp.

	Weights and biases are just references for MLP. We don't want to copy them!
	Optimizer means here Gradient Descent -optimizer. 
	There are several kinds of optimizers, and they have own constructors.
	
	(Calculating input-deltas are not necessary, and can save computation if skipped.)
	(But are needed if MLP's are connected to other MLP's. )
____________________________________________________________________________________________________
*/
#endregion


/// @func	___gradients_array_ini(mlp);
/// @desc	Initializes structure for previously created gradients-struct.
/// @param	{mlp_array}	mlp		Where structure is looked from
function ___gradients_array_ini(mlp) {
	trainingSession = 0;	// Divider for gradients to get average of several examples.

	momentumRate = 0;	// Default parameters for different optimizers.
	betaOne = .9;		// 
	betaTwo = .999;		// 
	iteration = 1;		//
	
	// Get structure information. For easy-to-access.
	layerCount = mlp.layerCount;
	layerSizes = array_create(layerCount, NULL);
	array_copy(layerSizes, 0, mlp.layerSizes, 0, layerCount);
	ActivationFunction = mlp.ActivationFunction;
	activation_parameter = mlp.activation_parameter;
		
	// Get references for easy-to-access. Don't make copy of these!
	activity	= mlp.activity;
	bias		= mlp.bias;
	output		= mlp.output;
	weights		= mlp.weights;	
		
	// Initialize deltas and gradients
	delta		= array_create(layerCount, NULL);	// Difference between actual output and wanted output. Consideres current example
	deltaSum	= array_create(layerCount, NULL);	// Cumulative delta from several examples
	gradients	= array_create(layerCount, NULL);	// First item is null.
	
	var i, j, jEnd, kEnd;
	delta[@0] = array_create(layerSizes[0], 0);
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		delta[@i] = array_create(jEnd, 0);
		deltaSum[@i] = array_create(jEnd, 0);
		gradients[@i] = array_create(jEnd, NULL);
		for(j = 0; j < jEnd; j++) {
			gradients[@i][@j] = array_create(kEnd, 0);
		}
	}
}


/// @func	___gradients_array_backpropagation();
/// @desc	Backpropagates output-error towards input-layer.
function ___gradients_array_backpropagation() {
	var i, j, k, jEnd, kEnd;
	var calculator, Derivative, parameter;

	// Backpropagate Delta through hidden layers.
	for(i = layerCount-1; i > 0; i--) {
		Derivative = ___DERIVATIVE[ActivationFunction[i]];
		parameter = activation_parameter[i];
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		calculator = Derivative(activity[i][j], parameter) * delta[i][j];
		deltaSum[@i][@j] += calculator;
		delta[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			gradients[@i][@j][@k] = gradients[i][j][k] + output[i-1][k] * calculator;
			delta[@i-1][@k] += weights[i][j][k] * calculator;
		}
	}}
	// This was 1 training session, add to count. 
	trainingSession++;
}

	
/// @func	___gradients_array_destroy();
/// @desc	Destroys gradients of optimizer. (Doesn't destroy structure of optimizer! )
function ___gradients_array_destroy() {
	// Destroy gradients and deltas
	delta = NULL;
	deltaSum = NULL;
	gradients = NULL;
		
	// Remove references.
	activity = NULL;
	bias = NULL;
	output = NULL;
	weights = NULL;
		
	// Left empty structure
	layerCount = 0;
	layerSizes = [0];
	trainingSession = 0;
}

