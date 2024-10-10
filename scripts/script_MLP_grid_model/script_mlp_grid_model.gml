#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	
____________________________________________________________________________________________________
*/
#endregion

/// @func	is_mlp_grid(mlp);
function is_mlp_grid(mlp) {
	return (instanceof(mlp) == script_get_name(mlp_grid));
}


/// @func	___mlp_grid_forward(inputArray);
function ___mlp_grid_forward(inputArray) {
	var i, j, k, jEnd, kEnd, J, K;
	var Activation, parameter;
	
	// Set input.
	jEnd = min(layerSizes[0], array_length(inputArray));
	array_copy(output[0], 0, inputArray, 0, jEnd);
		
	// Update values.
	for(i = 1; i < layerCount; i++) {
		Activation = ___ACTIVATION[ActivationFunction[i]];
		parameter = activation_parameter[i];
		jEnd = layerSizes[i];		J = jEnd-1;
		kEnd = layerSizes[i-1];		K = kEnd-1;		

		// Get weighted signals. 
		for(k = 0; k < kEnd; k++) {
			ds_grid_set_region(calculator, 0, k, J, k, output[i-1][k]);
		}	ds_grid_multiply_grid_region(calculator, weights[i], 0, 0, J, K, 0, 0);

		// Get combined signal, then calculate output. 
		for(j = 0; j < jEnd; j++) {
			activity[@i][@j] = ds_grid_get_sum(calculator, j, 0, j, K);
			output[@i][@j] = Activation(activity[i][j] + bias[i][j], parameter);
		}
	}
		
	// Return output. 
	return output[max(0, layerCount-1)];
}


/// @func	___mlp_grid_reset();
/// @desc	Resets MLP values with default random values.
function ___mlp_grid_reset() {
	var i, j, k, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		bias[@i][@j] = RandomBias();
	for(k = 0; k < kEnd; k++) {
		weights[@i][# j, k] = RandomWeight();
	}}}
}


/// @func	___mlp_grid_randomize(weightMin, weightMax, biasMin, biasMax);
/// @desc	Randomizes MLP values with given range.
function ___mlp_grid_randomize(weightMin, weightMax, biasMin, biasMax) {
	var i, j, k, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		bias[@i][@j] = random_range(biasMin, biasMax);
	for(k = 0; k < kEnd; k++) {
		weights[@i][# j, k] = random_range(weightMin, weightMax);
	}}}
}


/// @func	___mlp_grid_destroy();
/// @desc	Destroyes data-structures of MLP. This should be called when you delete MLP to avoid memory leaks.
function ___mlp_grid_destroy() {
	// Destroy possible optimizer
	OptimizerDestroy();
	
	// Destroy datastructures.		
	activity = NULL;
	output = NULL;
	bias = NULL;
	for(var i = 1; i < layerCount; i++) {
		ds_grid_destroy(weights[i]);
	}	weights = NULL;
	
	// Destroy calculator
	ds_grid_destroy(calculator);
	calculator = NULL;
	
	// Storing no layers.
	layerCount = 0;
	layerSizes = [0];
}

