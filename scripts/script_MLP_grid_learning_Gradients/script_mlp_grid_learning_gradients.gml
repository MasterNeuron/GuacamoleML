#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Creating gradients-structure from mlp. These don't need to be in mlp.

	Weights and biases are just references for MLP. We don't want to copy them!
	Optimizer means here Gradient Descent -optimizer. 
	There are several kinds of optimizers, and they have own constructors.
	
	(Calculating input-deltas are not necessary, and can save computation if skipped.)
	(But are needed if MLP's are connected to other MLP's. Currently they are skippted.)
____________________________________________________________________________________________________
*/
#endregion


/// @func	___gradients_grid_ini(mlp);
/// @desc	Initializes structure for previously created gradients-struct.
/// @param	{mlp_grid}	mlp			Where structure is looked from
function ___gradients_grid_ini(mlp) constructor {
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
	calculator = mlp.calculator;					// Use same calculator.

	// Get references for easy-to-access. Don't make copy of these!
	activity	= mlp.activity;
	bias		= mlp.bias;
	output		= mlp.output;
	weights		= mlp.weights;	
		
	// Initialize deltas and gradients
	delta		= array_create(layerCount, NULL);	// Difference between actual output and wanted output. Consideres current example
	deltaSum	= array_create(layerCount, NULL);	// Cumulative delta from several examples
	gradients	= array_create(layerCount, NULL);	// First item is null.

	var i, jEnd, kEnd;
	delta[@0] = array_create(layerSizes[0], 0);
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		delta[@i] = array_create(jEnd, 0);
		deltaSum[@i] = array_create(jEnd, 0);
		gradients[@i] = ds_grid_create(jEnd, kEnd);
	}
}
	

/// @func	___gradients_grid_backpropagation();
/// @desc	Backpropagates output-error towards input-layer.
function ___gradients_grid_backpropagation() {
	var i, j, k, jEnd, kEnd, J, K;
	var Derivative, parameter, deltaActivity;
	
	// Backpropagate Delta through hidden layers.
	for(i = layerCount-1; i > 0; i--) {
		Derivative = ___DERIVATIVE[ActivationFunction[i]];
		parameter = activation_parameter[i];
		jEnd = layerSizes[i];		J = jEnd-1;
		kEnd = layerSizes[i-1];		K = kEnd-1;
			
		// Finalize current deltas and calculate gradients
		for(j = 0; j < jEnd; j++) {
			deltaActivity = Derivative(activity[i][j], parameter) * delta[i][j];
			delta[@i][@j] = deltaActivity;
			deltaSum[@i][@j] = deltaSum[i][j] + deltaActivity;
			ds_grid_set_region(calculator, j, 0, j, K, deltaActivity);
		}

		// Calculate deltas for previous layer 
		ds_grid_multiply_grid_region(calculator, weights[i], 0, 0, J, K, 0, 0);
		for(k = 0; k < kEnd; k++) {
			delta[@i-1][@k] = ds_grid_get_sum(calculator, 0, k, J, k);
		}
	}
	
	// Calculate gradients
	for(i = layerCount-1; i > 0; i--) {
		jEnd = layerSizes[i];		J = jEnd-1;
		kEnd = layerSizes[i-1];		K = kEnd-1;

		for(j = 0; j < jEnd; j++) {
			ds_grid_set_region(calculator, j, 0, j, K, delta[i][j]);
		}
		for(k = 0; k < kEnd; k++) {
			ds_grid_multiply_region(calculator,	0, k, J, k, output[i-1][k]);
		}
		ds_grid_add_grid_region(gradients[i], calculator, 0, 0, J, K, 0, 0);
	}

	// This was 1 training session, add to count. 
	trainingSession++;
}


/// @func	___gradients_grid_destroy();
/// @desc	Destroys gradients of optimizer. (Doesn't destroy structure of optimizer! )
function ___gradients_grid_destroy() {
	// Destroy gradients and deltas
	delta = NULL;
	deltaSum = NULL;
	for(var i = 1; i < layerCount; i++) {
		ds_grid_destroy(gradients[i]);
	}	gradients = NULL;
		
	// Remove references.
	activity = NULL;
	bias = NULL;
	output = NULL;
	weights = NULL;
	calculator = NULL;

	// Left empty structure
	layerCount = 0;
	layerSizes = [0];
	trainingSession = 0;
}

