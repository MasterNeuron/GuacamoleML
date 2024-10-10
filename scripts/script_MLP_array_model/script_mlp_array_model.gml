#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	
____________________________________________________________________________________________________
*/
#endregion

/// @func	is_mlp_array(mlp);
function is_mlp_array(mlp) {
	return (instanceof(mlp) == script_get_name(mlp_array));
}


/// @func	___mlp_array_forward(inputArray);
function ___mlp_array_forward(inputArray) {
	var i, j, k, jEnd, kEnd, Activation, parameter;
	
	// Set input.
	jEnd = min(layerSizes[0], array_length(inputArray));
	array_copy(output[0], 0, inputArray, 0, jEnd);
		
	// Update values.
	for(i = 1; i < layerCount; i++) {
		Activation = ___ACTIVATION[ActivationFunction[i]];
		parameter = activation_parameter[i];
		jEnd = layerSizes[i];	
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		activity[@i][@j] = 0;
		for(k = 0; k < kEnd; k++) {
			activity[@i][@j] += output[i-1][k] * weights[i][j][k];
		}
		output[@i][@j] = Activation(activity[i][j] + bias[i][j], parameter);
	}}
		
	// Return output. 
	return output[max(0, layerCount-1)];
}


/// @func	___mlp_array_reset();
/// @desc	Resets MLP values with default random values.
function ___mlp_array_reset() {
	var i, j, k, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		bias[@i][@j] = RandomBias();
	for(k = 0; k < kEnd; k++) {
		weights[@i][@j][@k] = RandomWeight();
	}}}
}


/// @func	___mlp_array_randomize(weightMin, weightMax, biasMin, biasMax);
/// @desc	Randomizes MLP values with given range.
function ___mlp_array_randomize(weightMin, weightMax, biasMin, biasMax) {
	var i, j, k, jEnd, kEnd;
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
		bias[@i][@j] = random_range(biasMin, biasMax);
	for(k = 0; k < kEnd; k++) {
		weights[@i][@j][@k] = random_range(weightMin, weightMax);
	}}}
}


/// @func	___mlp_array_destroy();
/// @desc	Destroyes data-structures of MLP. This should be called when you delete MLP to avoid memory leaks.
function ___mlp_array_destroy() {
	// Destroy possible optimizer
	if (!is_undefined(optimizer)) {
		optimizer.Destroy();
	}	optimizer = NULL;
	
	// Destroy datastructures.
	activity = NULL;
	output	 = NULL;
	bias	 = NULL;
	weights  = NULL;
	
	// Storing no layers.
	layerCount = 0;
	layerSizes = [0];
}
