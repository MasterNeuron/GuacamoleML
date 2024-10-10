#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Multi-Layer Perceptron with arrays. 
	 - Layers are fully-connected. 
	 - As this is purely made with arrays etc., mlp is automatically garbage-collected.
	 - More readable than Grid-MLP, but slower.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_array(layerSizes, activationFunctions);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @desc	Use as constructor: "mlp = new mlp_array([4,8,2]);"
/// @param	{array}		layerSizes				Determine size of layers. First size is for input-layer, last for output-layer.
/// @param	{array}		activationFunctions		Activation functions for layers.
function mlp_array(layerSizeArray, activationFunctions) constructor {
	type = nn.ARRAY;
	
	// MLP variables.
	layerCount = 0;					// {Int}		How many layers there are, defined when struct is created
	layerSizes = [0];				// {1D Array}	How large layers are.
	activity = NULL;				// {2D Array}	Combined output signals from previous layers multiplied by linked weight. 
	bias = NULL;					// {2D Array}	Output signal, created with Activity + bias run in non-linear function
	output = NULL;					// {2D Array}	Bias for neuron activity
	weights = NULL;					// {3D Array}	Link between neurons.
	ActivationFunction = NULL;		// {1D Array}	Index for activation functions.
	activation_parameter = NULL;	// {1D Array}	(possible) Activation function parameters
	optimizer = NULL;				// {index}		Gradient descent optimizer.
	
	// Initialize structure.
	if (!is_undefined(layerSizeArray)) {
		___mlp_array_init(layerSizeArray, activationFunctions);
	}
	
	/// @func	Init(layerSizes);
	static Init = function(layerSizeArray, activationFunctions) {
		___mlp_array_init(layerSizeArray, activationFunctions);		
		return self;
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
	/// @func	Forward(inputArray);
	static Forward = function(inputArray) {
		return ___mlp_array_forward(inputArray);
	}
	/// @func	Output();	
	static Output = function() {
		return output[max(0, layerCount-1)];
	}
	/// @func	Draw(sprite, x, y, scale);
	static Draw = function(sprite, xPos, yPos, scale) {
		draw_mlp_array(self, sprite, xPos, yPos, scale);
	}
	/// @func	Reset();
	static Reset = function() {
		___mlp_array_reset();
		return self;
	}
	/// @func	Randomize(weightMin, weightMax, biasMin, biasMax);
	static Randomize = function(weightMin, weightMax, biasMin, biasMax) {
		___mlp_array_randomize(weightMin, weightMax, biasMin, biasMax);
		return self;
	}
	/// @func	Destroy();
	static Destroy = function() {
		___mlp_array_destroy();
	}
	/// @func	Copy(mlp);
	static Copy = function(mlp) {
		OptimizerDestroy()
		var buffer = mlp.Save();
		Load(buffer);
		buffer_delete(buffer);
		return self;
	}
	/// @func	Stringify();
	/// @desc	Returns MLP as JSON string
	static Stringify = function() {
		return ___mlp_array_stringify(self);
	}
	/// @func	Parse(jsonString);
	/// @desc	Loads values from JSON string
	static Parse = function(jsonString) {
		OptimizerDestroy()
		mlp_parse(self, jsonString);
		return self;
	}
	/// @func	Save(precision);
	/// @desc	Creates new buffer where MLP values are stored.
	static Save = function(precision) {
		return mlp_save_buffer(self, precision);
	}
	/// @func	Load(buffer);
	/// @desc	Loads MLP values from given buffer.
	static Load = function(buffer) {
		OptimizerDestroy()
		mlp_load_buffer(self, buffer);
		return self;
	}
}


/// @func	___mlp_array_init(layerSizes, activationFunctions);
/// @desc	Initializes previously created MLP. Sets layers neurons + weights by given layer-sizes.
/// @param	{mlp_grid}	mlp
/// @param	{array}		layerSizes
/// @param	{array}		activationFunctions
function ___mlp_array_init(layerSizeArray, activationFunctions) constructor {
	var i, j, k, jEnd, kEnd;
	if (!is_array(layerSizeArray)) {
		layerSizeArray = [];
	}
	
	// Inititialize layer-structure
	layerCount = array_length(layerSizeArray);
	layerSizes = array_create(layerCount, 0);
	array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);
	
	// Activation functions
	ActivationFunction = ___mlp_activation_functions(activationFunctions);
	activation_parameter = array_create(layerCount, .5);
	
	// Initialize neuron variables
	activity = array_create(layerCount, NULL);
	output	 = array_create(layerCount, NULL);
	bias	 = array_create(layerCount, NULL);
	weights	 = array_create(layerCount, NULL);
	if (layerCount == 0) return;

	// Initialize structure
	output[@0] = array_create(layerSizes[0], 0);	// Input-layer.
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i];
		kEnd = layerSizes[i-1];
		output[@i]		= array_create(jEnd, 0);
		bias[@i]		= array_create(jEnd, 0);
		activity[@i]	= array_create(jEnd, 0);
		weights[@i]		= array_create(jEnd, 0);
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] = RandomBias();
			weights[@i][@j] = array_create(kEnd, 0);
			for(k = 0; k < kEnd; k++) {
				weights[@i][@j][@k] = RandomWeight();
			}
		}
	}
}


