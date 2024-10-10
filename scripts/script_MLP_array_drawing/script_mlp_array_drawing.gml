#region /// INFORMATION ABOUT SCRIPT
/*
Functions to draw MLP's. Not prettiest ones, but atleast some visualization.
*/
#endregion

/// @func	draw_mlp_array(mlp, sprite, xStart, yStart, scale);
/// @desc	Draws MLP with 'snake' representation, showing individual weights*outputs clearly.
/// @param	{MLP}		mlp		neural network to be drawn
/// @param	{sprite}	sprite	For drawing neurons and weights
/// @param	{real}		xStart	
/// @param	{real}		yStart
/// @param	{real}		scale
function draw_mlp_array(mlp, sprite, xStart, yStart, scale) {
	
	var xx, yy, size, separation;
		size = sprite_get_width(sprite);
		separation = size * 2;

	var dir, pos, xPos, yPos, color;
		dir = [-90,0,90,0];
		pos = [0,1,0,1];
		
	// Draw neurons
	xx = xStart;
	yy = yStart;
	
	for(var i = 0; i < mlp.layerCount; i++) {
		matrix_start(xx, yy, scale, scale, dir[i mod 4]);
		for(var j = 0; j < mlp.layerSizes[i]; j++) {
			color = NeuronColor(mlp.output[i][j]);
			xPos = (separation * pos[(i+1) mod 4] + j * size);
			yPos = 0;
			draw_sprite_ext(sprite,0,xPos,yPos,1,2,0,color,1);
		}
		xx += (separation + pos[i mod 4] * size * (mlp.layerSizes[i]-1)) * scale;
	}
	matrix_end();
	
	
	// Draw weights
	var flip, xPos, yPos;
		dir = [-90,0,90,0];
		pos = [0,1,0,1];
		flip = [1,1,-1,-1];
		xx = xStart + (separation + pos[i mod 4] * size - size) * scale;
		yy = yStart;
	
	for(var i = 1; i < mlp.layerCount; i++) {
	
		matrix_start(xx, yy, scale, scale, dir[i mod 4]);
		for(var j = 0; j < mlp.layerSizes[i]; j++) {
			for(var k = 0; k < mlp.layerSizes[i-1]; k++) {
			
				xPos = (separation * pos[(i+1) mod 4] + j * size);
				yPos = (separation + k * size) * flip[i mod 4];
				//color = WeightColor(mlp.weights[i][j][k]);
				color = WeightColor(mlp.weights[i][j][k] * mlp.output[i-1][k]);
				//color = WeightColor(mlp.gradients[i][j][k]); // To show current gradient
				draw_sprite_ext(sprite,0, xPos, yPos, 1,1,0,color,1);
			}
		}
		xx += (separation + pos[i mod 4] * size * (mlp.layerSizes[i]-1)) * scale;
	}
	matrix_end();
}
