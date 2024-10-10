
/// @func	 ___optimizer_plus(mlp, EnumOptimizer, ?momentumRate);
/// @desc	Creates gradient descent optimizer for given mlp.
/// @param	{mlp_plus}	mlp
/// @param	{enum}		EnumOptimizer
/// @param	{real}		?momentumRate 
function ___optimizer_plus(mlp, EnumOptimizer, momentumRate) constructor {
	
	// Initialize optimzier
	trainingSession	= 0;											// For user interface, making this behave similiarly to other MLP-types. 
	optimizer_index = ___ext_optimizer_create(mlp.mlp_index, EnumOptimizer);
	inputSize		= mlp.inputSize;
	outputSize		= mlp.outputSize;
	targetBuffer	= buffer_create(outputSize*8, buffer_wrap, 8);	// For calculating output-error. Send to extension, calculates outputBuffer.
	outputBuffer	= buffer_create(outputSize*8, buffer_wrap, 8);	// Output delta. For using own cost-functions in GML.
	inputBuffer		= buffer_create(inputSize*8, buffer_wrap, 8);	// Input delta. For chaining mlp's backpropagations
	inputDelta		= array_create(inputSize);						// For returning backpropagated delta result. This can be used for linking MLP's together.

	___ext_optimizer_buffer_set(optimizer_index, 
		buffer_get_address(targetBuffer), 
		buffer_get_address(outputBuffer), 
		buffer_get_address(inputBuffer));
	
	// Set default values. Not all use these.
	momentumRate = is_undefined(momentumRate) ? 0 : momentumRate;	// momentum- and decayrates here.
	var betaOne = .9;		// Adam and some of it's variations use
	var betaTwo = .999;		// 
	___ext_optimizer_defaults(optimizer_index, momentumRate, betaOne, betaTwo);

	// METHODS
		
	/// @func	Backward(); 
	/// @desc	Backpropagation. Updates delta-errors, moving through from output towards input layer.
	static Backward = function() {
		___ext_optimizer_backpropagate(optimizer_index);
		trainingSession++;
		return ___ext_optimizer_input_delta();
	}
	
	/// @func	Cost(costFunction, target, ?parameter);
	/// @desc	Calculates error for current output, returns total error. Updates delta-error for output-layer.
	/// @param	{function}	costFunction	
	/// @param	{array}		target			MLP values are compared against given target values
	/// @param	{real}		?parameter		Optional hyperparameter for cost-function.
	static Cost = function(costFunction, target, hyperParameter) {
		___optimizer_plus_set_target(target);
		return ___ext_optimizer_cost(optimizer_index, cost_function_enum(costFunction), hyperParameter);
	}
	
	/// @func	Delta(target);
	/// @desc	Sets output-delta error to given delta-array. Used for linking MLP's together.
	/// @param	{array}		target
	static Delta = function(deltaTarget) {
		___optimizer_plus_set_target(deltaTarget);
		return ___ext_optimizer_cost(optimizer_index, cost_function_enum(Delta), 0);
	}	
	
	/// @func	Decay(rate);
	/// @desc	Simple weight decaying function. Use this after "Apply()"
	/// @param	{real}	rate	How fast decays. For example ".01".	
	static Decay = function(rate) {
		___ext_optimizer_weight_decay(optimizer_index, rate);
	}

	/// @func	Apply(learnRate);
	/// @desc	Applies average gradients to weights of MLP.
	static Apply = function(learnRate) {
		___ext_optimizer_apply(optimizer_index, learnRate);
		trainingSession = 0;
	}

	/// @func	Destroy();
	/// @desc	Destroyes optimizer.
	static Destroy = function() {
		___ext_optimizer_destroy(optimizer_index);
		buffer_delete(targetBuffer);
		buffer_delete(outputBuffer);
		buffer_delete(inputBuffer);
	}
}

/// @func	___optimizer_plus_set_target(targetArray);
/// @desc	Set target values for calling optimizer, which are used to update mlp.
/// @param	{array}		targetArray
function ___optimizer_plus_set_target(targetArray) {
	buffer_seek(targetBuffer, buffer_seek_start, 0);
	for(var i = 0; i < outputSize; i++) {
		buffer_write(targetBuffer, NumberType.DOUBLE, targetArray[i]);
	}
}

/// @func	___ext_optimizer_input_delta();
/// @desc	Updates input delta-array using buffer, which stores delta and is being updated by extension.
function ___ext_optimizer_input_delta() {
	buffer_seek(inputBuffer, buffer_seek_start, 0);
	for(var i = 0; i < inputSize; i++) {
		input[i] = buffer_read(inputBuffer, NumberType.DOUBLE);
	}
	return input;
}
