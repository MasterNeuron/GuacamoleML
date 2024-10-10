#macro	NULL	undefined
#macro	PRINT	show_debug_message

enum nn {
	ARRAY, GRID, PLUS
}
enum NumberType {
	HALF = buffer_f16,
	FLOAT = buffer_f32,
	DOUBLE = buffer_f64
}
enum SizeOf {
	HALF = 2,
	FLOAT = 4,
	DOUBLE = 8
}
