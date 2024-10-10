#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	This script considers how you load and save MLP with buffers. 
	File-size is smaller than JSON, but isn't human-readable. 
	Training methods are not saved, need to be renitialized.
	Lower precision reduces file-size. But this might cause side-effects.
	Precision is stored as GML buffer-constant,		buffer_f64: 9.		buffer_f32: 8.		buffer_f16: 7.
____________________________________________________________________________________________________
				
	BUFFER FORMAT:
	
	METADATA
		Precision			: 8 bytes. How precisely weight and biases are stored.
		layerCount			: 8 bytes
		layerSize			: 8 bytes
			...					-> Repeated by layerCount.
		ActivationFunction	: 8 bytes. Stored as enum-index
			...					-> Repeated by layerCount.
		activation_params.	: 8 bytes. Possible Activation function parameters.
			...					-> Repeated by layerCount. 
	
	PARAMETERS
		Biases				: 2-8 bytes, float-point number. 
			...					-> Repeated by sum of layerSizes. (without input-layer)
		Weights				: 2-8 bytes, float-point number. 
			...					-> Repeated by sum of layerSize multiples (layerSize * previous LayerSize).
____________________________________________________________________________________________________
*/
#endregion

/*____________________________________________________________________________________________________
*/
	
/// @func	mlp_save_buffer(mlp, precision);
/// @desc	Creates buffer and saves values of MLP to it. You can determine how accurately float numbers are stored.
/// @param	{mlp}		mlp	
/// @param	{enum}		precision	How precicely numbers are stored. Uses buffer -constants: buffer_f64, buffer_f32, buffer_f16.
/// @return	{buffer}	mlpBuffer	Returns	index for buffer. Buffer type is 'fixed'
function mlp_save_buffer(mlp, precision) {
	precision = ___mlp_acceptable_precision(precision);
	
	// ACTION DEPENT ON MLP-TYPE
	switch(mlp.type) {
		case nn.ARRAY:	var get_weight = function(weights, i, j, k) { return weights[i][j][k]; }	break;
		case nn.GRID:	var get_weight = function(weights, i, j, k) { return weights[i][# j, k]; }	break;
		case nn.PLUS:	return ___mlp_plus_save_buffer(mlp, precision);
		default: throw("Unknown MLP-type in 'mlp_save_buffer'.");
	}
	var i, j, k, jEnd, kEnd;

	// CREATE BUFFER
	with(mlp) { 
	var sizeOf = buffer_sizeof(NumberType.DOUBLE);
	var bufferSize = 0;
	bufferSize += sizeOf * 2;									// Precision & layerCount
	bufferSize += sizeOf * array_length(layerSizes) * 3;		// Layer-sizes, Activation-functions & -parameters
	sizeOf = buffer_sizeof(precision);							//
	for(i = 1; i < layerCount; i++) {							//
		bufferSize += sizeOf * layerSizes[i];					// Biases
		bufferSize += sizeOf * layerSizes[i] * layerSizes[i-1];	// Weights
	}															//
	var buffer = buffer_create(bufferSize, buffer_fixed, 1);	//
		
	// Write meta-data
	buffer_write(buffer, NumberType.DOUBLE, precision);
	buffer_write(buffer, NumberType.DOUBLE, layerCount);
	for(i = 0; i < layerCount; i++) {
		buffer_write(buffer, NumberType.DOUBLE, layerSizes[i]);
	}
	for(i = 0; i < layerCount; i++) {
		buffer_write(buffer, NumberType.DOUBLE, ActivationFunction[i]);
	}
	for(i = 0; i < layerCount; i++) {
		buffer_write(buffer, NumberType.DOUBLE, activation_parameter[i]);
	}
	
	// Write biases
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
	for(j = 0; j < jEnd; j++) {
		buffer_write(buffer, precision, bias[i][j]);
	}}

	// Write weights
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
	for(k = 0; k < kEnd; k++) {
		buffer_write(buffer, precision, get_weight(weights, i, j, k));
		
	}}}}
	
	// RETURN BUFFER
	buffer_seek(buffer, buffer_seek_start, 0);
	return buffer;	
}

/*____________________________________________________________________________________________________
*/

/// @func	mlp_load_buffer(mlp, buffer);
/// @desc	Loads values to MLP from buffer, which is previously created with mlp_save_buffer().
/// @param	{mlp}		mlp			Target MLP where values are loaded
/// @param	{buffer}	buffer
function mlp_load_buffer(mlp, buffer) {	
	// ACTION DEPENT ON MLP-TYPE
	switch(mlp.type) {
		case nn.ARRAY:	var set_weight = function(mlp, i, j, k, value) { mlp.weights[@i][@j][@k] = value;	}	break;
		case nn.GRID:	var set_weight = function(mlp, i, j, k, value) { mlp.weights[i][# j, k] = value;	}	break;
		case nn.PLUS:	return ___mlp_plus_load_buffer(mlp, buffer);
		default: throw("Unknown MLP-type in 'mlp_load_buffer'.");
	}
		
	// Read meta-data
	var i, j, k, jEnd, kEnd;
	buffer_seek(buffer, buffer_seek_start, 0);
	var precision	= ___mlp_acceptable_precision(buffer_read(buffer, NumberType.DOUBLE));
	var layerCount	= buffer_read(buffer, NumberType.DOUBLE);
	var layerSizes	= array_create(layerCount, 0);
	for(i = 0; i < layerCount; i++) {
		layerSizes[i] = buffer_read(buffer, NumberType.DOUBLE);
	}
	var activationFunctions = array_create(layerCount, 0);
	for(i = 0; i < layerCount; i++) {
		activationFunctions[i] = buffer_read(buffer, NumberType.DOUBLE);
	}
	var activation_parameter = array_create(layerCount, 0);
	for(i = 0; i < layerCount; i++) {
		activation_parameter[i] = buffer_read(buffer, NumberType.DOUBLE);
	}

	// Reinitialize target mlp.
	mlp.Init(layerSizes);
	array_copy(mlp.ActivationFunction, 0, activationFunctions, 0, layerCount);
	array_copy(mlp.activation_parameter, 0, activation_parameter, 0, layerCount);
	
	// Read biases
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
	for(j = 0; j < jEnd; j++) {
		mlp.bias[@i][@j] = buffer_read(buffer, precision);
	}}

	// Read weights
	for(i = 1; i < layerCount; i++) {
		jEnd = layerSizes[i]; 
		kEnd = layerSizes[i-1];
	for(j = 0; j < jEnd; j++) {
	for(k = 0; k < kEnd; k++) {
		set_weight(mlp, i, j, k, buffer_read(buffer, precision));
	}}}
	buffer_seek(buffer, buffer_seek_start, 0);
}
	
/*____________________________________________________________________________________________________
*/

/// @func	___mlp_plus_save_buffer(mlp, precision);
/// @desc	Saves mlp to buffer. 
/// @param	{mlp_plus}	mlp
function ___mlp_plus_save_buffer(mlp, precision) {
	var buffer, bufferSize;
	precision = ___mlp_acceptable_precision(precision);
	bufferSize = ___ext_mlp_size_export(mlp.mlp_index, precision);
	buffer = buffer_create(bufferSize, buffer_fixed, 1);
	___ext_mlp_export(mlp.mlp_index, precision, buffer_get_address(buffer));
	
	// This new buffer might not work correctly as GMS might get confused, as GMS hasn't itself done anything with it (only extension)
	// (eg. buffer_save doesn't work, saves only empty file)
	// So this little trick is done to make sure GMS "knows" buffer has something, and it behaves correctly!
	buffer_poke(buffer, bufferSize-1, buffer_u8, buffer_peek(buffer, bufferSize-1, buffer_u8));
	// Write and read last item, so GMS knows buffer has something up until that point.
	// Eg. If done to item in middle, and used buffer_save(...), GMS will only save buffer until that point.

	buffer_seek(buffer, buffer_seek_start, 0);	// just make sure buffer tell is in start
	return buffer;
}

/// @func	___mlp_plus_load_buffer(mlp, mlpBuffer);
/// @desc	Loads buffer to target mlp.
/// @param	{mlp_plus}	mlp
/// @param	{buffer}	buffer	Has mlp byte-encoded.
function ___mlp_plus_load_buffer(mlp, buffer) {
	with(mlp) {
	___ext_mlp_destroy(mlp_index);
	mlp_index = ___ext_mlp_import(
		buffer_get_address(buffer));
	inputSize = ___ext_mlp_size_input(mlp_index);
	outputSize = ___ext_mlp_size_output(mlp_index);
	buffer_resize(inputBuffer, inputSize*8);
	buffer_resize(outputBuffer, outputSize*8);
	array_resize(outputArray, outputSize);
	___ext_mlp_buffers_set(mlp_index, 
		buffer_get_address(inputBuffer),
		buffer_get_address(outputBuffer));
	buffer_seek(buffer, buffer_seek_start, 0);
	return mlp_index;
}}
