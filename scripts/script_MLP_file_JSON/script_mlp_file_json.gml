#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	This script is about loading and saving MLP. 
	You can stringify MLP as JSON-string or text-file, which you can parse later.
	Training methods are not saved, need to be Initialized again after Copy or Loading.
	Stringifying and parsing is bit inefficient, but made this way to avoid repeating code.
	Stringifying and parsing happens through buffer and array-mlp, as it's easiest.
____________________________________________________________________________________________________
*/
#endregion
	
	
/// @func	mlp_parse(mlp, jsonString);
/// @desc	Imports values from JSON string to previously created MLP.
/// @param	{mlp_array}	mlp			target MLP where values are copied to.
/// @param	{string}	jsonString
function mlp_parse(mlp, jsonString) {
	var dummy = json_parse(jsonString);
		dummy.type = nn.ARRAY;		// Take advantage of struct being similiar to mlp-array.
	var buffer = mlp_save_buffer(dummy, NumberType.DOUBLE);	// Translate to buffer. 
	mlp.Load(buffer);
	buffer_delete(buffer);
	delete dummy;
}
	
	
/// @func	mlp_stringify(mlp);
/// @desc	Stringifies given Grid and Plus-mlp's.  
/// @desc	Inefficient way of doing this, but simplest. Also reduces code that needs to be maintained.
function mlp_stringify(mlp) {
	// Easiest if Array.
	if (mlp.type == nn.ARRAY) {
		return ___mlp_array_stringify(mlp);
	}
	// Otherwise make dummy which is stringified.
	var buffer, dummy, jsonString;
	buffer = mlp.Save();
	dummy = new mlp_array().Load(buffer);
	jsonString = dummy.Stringify();
	dummy.Destroy();
	delete dummy;
	return jsonString;
}



/*____________________________________________________________________________________________________
*/

/// @func	___mlp_array_stringify(mlp);
/// @desc	Turns target MLP's important values to JSON, and returns it as string.
/// @param	{mlp_array}	mlp
/// @return	{string}	json
function ___mlp_array_stringify(mlp) {		
	with(mlp) {
	var dummy, jsonString;			// Create dummy struct, where only necessary are stored
	dummy = {};						// Array-mlp is easy to stringify, just take references to important arrays and values.	
	dummy.layerCount = layerCount;		
	dummy.layerSizes = layerSizes;
	dummy.ActivationFunction = ActivationFunction;		// For actual parsing, stores only enum-indexes.
	dummy.activation_parameter = activation_parameter;
	dummy.activation_functions = activation_function_list_array(ActivationFunction);	// For JSON readibility. Not used while parsing.
	dummy.bias = bias;
	dummy.weights = weights;
	jsonString = json_stringify(dummy);
	delete dummy;
	return jsonString;
}}
