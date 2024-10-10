
/// @func	mlp_plus_test_linkage();
/// @desc	Tests wether Plus is available.
function mlp_plus_test_linkage() {
	return (os_type == os_windows) && (___ext_mlp_test_linkage());
}

/// @func	mlp_plus_forward(mlp, inputArray);
/// @desc	Updates mlp output with given input
/// @param	{mlp_plus}	mlp
/// @param	{array}	inputArray
function mlp_plus_forward(mlp, inputArray) {
	mlp_plus_input(mlp, inputArray);
	return ___ext_mlp_forward(mlp.mlp_index);	
}

/// @func	mlp_plus_input(mlp, inputArray);
/// @desc	Returns output as array
/// @param	{mlp_plus}	mlp
/// @param	{array}		inputArray
function mlp_plus_input(mlp, inputArray) {
	with(mlp) {
	buffer_seek(inputBuffer, buffer_seek_start, 0);
	for(var i = 0; i < inputSize; i++) {
		buffer_write(inputBuffer, NumberType.DOUBLE, inputArray[i]);
	}
}}

/// @func	mlp_plus_output(mlp);
/// @desc	Returns output as array. You can also read output directly from buffer.
/// @param	{mlp_plus}	mlp
function mlp_plus_output(mlp) {
	with(mlp) {
	buffer_seek(outputBuffer, buffer_seek_start, 0);
	for(var i = 0; i < outputSize; i++) {
		outputArray[i] = buffer_read(outputBuffer, NumberType.DOUBLE);
	}
	return outputArray;
}}

/// @func	mlp_plus_output_pos(mlp);
/// @desc	Returns single output from given mlp
/// @param	{mlp_plus}	mlp
function mlp_plus_output_pos(mlp, index) {
	return buffer_peek(mlp.outputBuffer, index*8, NumberType.DOUBLE);
}


/// @func	mlp_plus_copy(mlp, mlpTarget);
/// @desc	Copies values from another MLP. 
/// @param	{mlp_plus}	mlp
/// @param	{mlp}		mlpTarget
function mlp_plus_copy(mlp, mlpTarget) {
	// Easiest way is to translate other to buffer.
	var buffer = mlpTarget.Save(NumberType.DOUBLE);
	
	// Reinitialize mlp
	with(mlp) {
	___ext_mlp_destroy(mlp_index);
	mlp_index = ___ext_mlp_import(buffer_get_address(buffer));
	inputSize = ___ext_mlp_size_input(mlp_index);
	outputSize = ___ext_mlp_size_output(mlp_index);
	buffer_resize(inputBuffer, inputSize*8);
	buffer_resize(outputBuffer, outputSize*8);
	array_resize(outputArray, outputSize);
	___ext_mlp_buffers_set(mlp_index, 
		buffer_get_address(inputBuffer), 
		buffer_get_address(outputBuffer));
	}
	buffer_delete(buffer);
}


