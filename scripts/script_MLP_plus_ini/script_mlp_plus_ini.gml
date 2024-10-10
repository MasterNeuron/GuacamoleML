#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Magic number "8" in buffers is for memory size: 8 bytes, which comes from 64bit-float point number.
____________________________________________________________________________________________________
*/
#endregion



/// @func	mlp_plus(layerSizes, activationFunctions);
/// @desc	Creates mlp, which uses extension.
/// @param	{array}	layerSizeArray
/// @param	{array}	activationFunctions
function mlp_plus(layerSizeArray, activationFunctions) constructor {
	// Initialize values.
	type = nn.PLUS;
	layerCount = 0;
	inputSize = 0; 
	outputSize = 0;
	inputBuffer = buffer_create(8, buffer_wrap, 8);
	outputBuffer = buffer_create(8, buffer_wrap, 8);
	outputArray = [];
	optimizer = NULL;
	
	// Create mlp inside extension. Define variables.
	mlp_index = ___mlp_plus_ini(layerSizeArray, activationFunctions);

	/// @func	Init(layerSizes, activationFunctions);
	static Init = function(layerSizeArray, activationFunctions) {
		// Reinitalize mlp.
		___ext_mlp_destroy(mlp_index);
		mlp_index = ___mlp_plus_ini(layerSizeArray, activationFunctions);
				
		// Optimizer is now incorrect.
		OptimizerDestroy();
		return self;
	}

	/// @func	Forward(inputArray);
	static Forward = function(inputArray) {
		mlp_plus_forward(self, inputArray);
		return mlp_plus_output(self);
	}
	/// @func	Output();	
	static Output = function() {
		return mlp_plus_output(self);
	}
	/// @func	Optimizer(GradientDescent, ?parameter);
	/// @desc	Creates gradient-descent optimizer for mlp. Stored in mlp.
	/// @param	{wrapper}	GradientDescent		Give index of GradientDescent-optimizer wrapper (Stochastic, Momentum, Nesterov, Adam).
	/// @param	{real}		?parameter			Optional argument constructor can use.
	static Optimizer = function(gradientDescent, parameter) {
		OptimizerDestroy();
		optimizer = gradientDescent(parameter);
		return self;
	}
	/// @func	OptimizerDestroy();
	static OptimizerDestroy = function() {
		if (!is_undefined(optimizer)) {
			optimizer.Destroy();
			optimizer = NULL;
		}
	}
	/// @func	Draw(sprite, x, y, scale);
	static Draw = function(sprite, xPos, yPos, scale) {
		draw_mlp_plus(self, sprite, xPos, yPos, scale);
	}
	/// @func	Reset();
	static Reset = function() {
		___ext_mlp_randomize(mlp_index, -.5, +.5, -.2, +.2);
	}
	/// @func	Randomize(weightMin, weightMax, biasMin, biasMax);
	static Randomize = function(weightMin, weightMax, biasMin, biasMax) {
		___ext_mlp_randomize(mlp_index, weightMin, weightMax, biasMin, biasMax);
	}
	/// @func	Destroy();
	static Destroy = function() {
		// Destroy optimizer
		OptimizerDestroy();
		// Destroy datastructure
		___ext_mlp_destroy(mlp_index);
		buffer_delete(inputBuffer);
		buffer_delete(outputBuffer);
		outputArray = NULL;
	}
	/// @func	Copy(mlp);
	static Copy = function(mlp) {
		OptimizerDestroy();
		mlp_plus_copy(self, mlp);
		return self;
	}
	/// @func	Stringify();
	/// @desc	Return MLP values as JSON-string.
	static Stringify = function() {
		return mlp_stringify(self);
	}
	/// @func	Parse(jsonString);
	/// @desc	Loads MLP values from given JSON-string.
	static Parse = function(jsonString) {
		OptimizerDestroy();
		mlp_parse(self, jsonString)
		return self;
	}		
	/// @func	Save(precision);
	/// @desc	Creates new buffer where MLP values are stored.
	static Save = function(precision) {
		return ___mlp_plus_save_buffer(self, precision);
	}
	/// @func	Load(buffer);
	/// @desc	Loads MLP values from given buffer.
	static Load = function(buffer) {
		OptimizerDestroy();
		return ___mlp_plus_load_buffer(self, buffer);
	}	
}

/// @func	___mlp_plus_ini(layerSizes, activationFunctions);
/// @desc	Creates new mlp in extension with given layers, returns index for it.
/// @param	{array}	layerSizes
/// @param	{array}	activationFunctions
function ___mlp_plus_ini(layerSizesArray, activationFunctions) {
	var mlp_index, i, iEnd, debugSuccess;
	
	// Set minimium size [1,1], otherwise extension wont like.
	if (is_undefined(layerSizesArray))
	or (!is_array(layerSizesArray))
	or (array_length(layerSizesArray) == 0) {
		layerSizesArray = [1, 1];
	} else if (array_length(layerSizesArray) == 1) {
		layerSizesArray[1] = 1;
	}
	// Activation functions
	layerCount = array_length(layerSizesArray);
	activationFunctions = ___mlp_activation_functions(activationFunctions);

	// Create mlp in extension
	mlp_index = ___ext_mlp_create();
	iEnd = array_length(layerSizesArray);
	for(i = 0; i < iEnd; i++) {							
		debugSuccess = ___ext_mlp_add_layer(mlp_index, layerSizesArray[i]);
		debugSuccess = ___ext_mlp_add_activation(mlp_index, activationFunctions[i]);
	}
	debugSuccess = ___ext_mlp_build(mlp_index);	

	// Set struct data.
	inputSize = ___ext_mlp_size_input(mlp_index);
	outputSize = ___ext_mlp_size_output(mlp_index);
	buffer_resize(inputBuffer, inputSize*8);
	buffer_resize(outputBuffer, outputSize*8);
	debugSuccess = ___ext_mlp_buffers_set(mlp_index, 
		buffer_get_address(inputBuffer), 
		buffer_get_address(outputBuffer));
	array_resize(outputArray, outputSize);

	return mlp_index;
}