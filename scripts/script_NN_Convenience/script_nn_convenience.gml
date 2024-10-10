#region /// INFORMATION ABOUT SCRIPT
/*
	Miscellaneous functions for different purposes. Most likely convenience.
*/
#endregion



/// @func	todo(text);
/// @desc	Easy to search, things written what needs to be done
function todo(text) {
	return;	// Doesn't have any functionality though.
}

/// @func	matrix_start(x,y,xscale,yscale,rotation);
/// @desc	For convenience. Drawing thing.
function matrix_start(_x,_y,_xscale,_yscale,_rotation) {
	matrix_set( matrix_world, matrix_build( _x, _y, 0, 0, 0, _rotation, _xscale, _yscale, 1));
}

/// @func	matrix_end();
/// @desc	For convenience, Drawing thing.
function matrix_end() {
	matrix_set( matrix_world, matrix_build_identity());
}

/// @func	NeuronColor(value);
/// @desc	Returns color for given value. Used for drawing neural networks.
/// @param	{real}	value	between -1 and +1
/// @return	{color}
function NeuronColor(value) {
	value		= Tanh(value);
	var red		= (abs(value) - value) * 111+16;
	var green	= (abs(value) + value) * 111+16;
	var blue	= (abs(value)) * 111 + 32;
	return make_color_rgb( red, green, blue);
} 

/// @func	WeightColor(value);
/// @desc	Returns color for given value. Used for drawing neural networks.
/// @param	{real}	value
/// @return	{color}
function WeightColor(value) {
	var overValue	= clamp((value-1)*64,0,192);
	value			= Tanh(value);	// Squishes value to be between -1 and +1
	var hue			= (value + 1) * 48-16;
	var saturation	= 192 - abs(value) * 64 - overValue;
	var brightness	= 16 + abs(value) * 127 + overValue/3;
	return make_color_hsv( hue, saturation, brightness);
} 

/// @func	approximate_derivative(function, input);
/// @desc	For debugging. This function approximates derivative. Only 1 input function.
/// @desc	This is used for testing wether hand-written derivatives behaves correctly
/// @param	{function}	function	
/// @param	{real}		input		Returns derivative value for given input.
function approximate_derivative(func, input, indexDerivative) {
	var epsilon = .00001;
	return (func(input + epsilon) - func(input)) / epsilon;
}

/// @func	approximate_derivative_array(function, inputs, index);
/// @desc	For debugging. This function approximates derivative. This is used for testing wether hand-written derivatives behaves correctly
/// @desc	Takes function arguments as input-array. Also give index which input's derivative wanted.
/// @param	{function}	function	
/// @param	{array}		inputs		Input array, so multi-argument functions are supported
/// @param	{int}		index		Index for which input we look derivative for.
function approximate_derivative_array(func, inputs, indexDerivative) {
	var epsilon = .00001;
	var value1 = script_execute_ext(func, inputs);
	if (is_array(inputs[indexDerivative])) {
		inputs[indexDerivative][0] += epsilon;
	} else {
		inputs[indexDerivative] += epsilon;
	}
	var value2 = script_execute_ext(func, inputs);
	return (value2 - value1) / epsilon;
}