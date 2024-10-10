#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Multi-Layer Perceptron with ds_grids. 
	 - As layers are fully-connected, weights make 'rectangular shape'. Therefore it's easy to store weights as ds_grids. 
	 - Compared to GML array, this gives speed boost, which is important especially when training.
	 - As downside you need to explicitly destroy data-structures.
	 - Still slower than using extension.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_grid(layerSizes, activationFunctions);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @desc	Use as constructor: "mlp = new mlp_grid([4,8,2], Tanh);"
/// @param	{array}		layerSizes				Determine size of layers. First size is for input-layer, last for output-layer.
/// @param	{array}		activationFunctions		Activation functions for layers.
function mlp_grid(layerSizeArray, activationFunctions) constructor {
	type = nn.GRID;
	
	// MLP variables.
	layerCount = 0;					// {Int}		How many layers there are, defined when struct is created
	layerSizes = [0];				// {1D Array}	How large layers are.
	activity = NULL;				// {2D Array}	Combined output signals from previous layers multiplied by linked weight. 
	bias = NULL;					// {2D Array}	Output signal, created with Activity + bias run in non-linear function
	output = NULL;					// {2D Array}	Bias for neuron activity
	weights = NULL;					// {3D Grid}	Link between neurons.
	ActivationFunction = NULL;		// {1D Array}	Index for activation functions.
	activation_parameter = NULL;	// {1D Array}	(possible) Activation function parameters
	optimizer = NULL;				// {index}		Gradient descent optimizer.
	
	// Calculator
	calculator = NULL;		// Calculations need other grid. Having own static reduces constant creating/destroying of temporal grids.
							// Also grid is created with maximium layer-dimension, so don't have to resize it constantly.

	// Initialize structure, if defined
	if (!is_undefined(layerSizeArray)) {
		___mlp_grid_init(layerSizeArray, activationFunctions);
	}
	
	/// @func	Init(layerSizes, activationFunctions);
	static Init = function(layerSizeArray, activationFunctions) {
		___mlp_grid_init(layerSizeArray, activationFunctions);
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
		return ___mlp_grid_forward(inputArray);
	}
	/// @func	Output();	
	static Output = function() {
		return output[max(0, layerCount-1)];
	}
	/// @func	Draw(sprite, x, y, scale);
	static Draw = function(sprite, xPos, yPos, scale) {
		draw_mlp_grid(self, sprite, xPos, yPos, scale);
	}
	/// @func	Reset();
	static Reset = function() {
		___mlp_grid_reset();
		return self;
	}
	/// @func	Randomize(weightMin, weightMax, biasMin, biasMax);
	static Randomize = function(weightMin, weightMax, biasMin, biasMax) {
		___mlp_grid_randomize(weightMin, weightMax, biasMin, biasMax);
		return self;
	}
	/// @func	Destroy();
	static Destroy = function() {
		___mlp_grid_destroy();
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
		return mlp_stringify(self);
	}
	/// @func	Parse(jsonString);
	/// @desc	Loads values from JSON string
	static Parse = function(jsonString) {
		OptimizerDestroy();
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
		OptimizerDestroy();
		mlp_load_buffer(self, buffer);
		return self;
	}
}

/// @func	___mlp_grid_init(layerSizes, activationFunctions);
/// @desc	Initializes previously created MLP. Sets layers neurons + weights by given layer-sizes.
/// @param	{mlp_grid}	mlp
/// @param	{array}		layerSizes
/// @param	{array}		activationFunctions
function ___mlp_grid_init(layerSizeArray, activationFunctions) constructor {
	var i, j, k, jEnd, kEnd;
	if (!is_array(layerSizeArray)) {
		layerSizeArray = [];
	}
		
	// Inititialize layer-structure
	layerCount = array_length(layerSizeArray);
	layerSizes = array_create(layerCount, 0);
	array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);
		
	// Activation funtions
	ActivationFunction = ___mlp_activation_functions(activationFunctions);
	activation_parameter = array_create(layerCount, .5);

	// Create helper calculator
	var maxSize = 0;
	for(i = 0; i < layerCount; i++) {				// Used for grid-calculations.
		maxSize = max(maxSize, layerSizes[i]);		// To avoid constant creation/destruction/resizing of temporal grids.
	}	calculator = ds_grid_create(maxSize, maxSize);
		
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
		weights[@i]		= ds_grid_create(jEnd, kEnd);
		for(j = 0; j < jEnd; j++) {
			bias[@i][@j] = RandomBias();
		for(k = 0; k < kEnd; k++) {
			weights[@i][# j, k] = RandomWeight();
		}}
	}
}


