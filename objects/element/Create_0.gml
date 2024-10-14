// SIMULATIONAL RATING
rating = 0;
// TOP ELEMENT
top = 0;
// CREATE THE MLP - DROP THE BEAT!
mlp = new mlp_grid([8, 64, 32, 16, 8, 1]);
// RANDOMIZE THE FIRST VALUES
mlp.Randomize(-1, +1, -1, +1);

// CREATE THE MAIN RUN FUNCTION
Run = function()
	{
// GET THE PRICE
price = global.price[global.simstep];

// FOR THE TEST
//input[0] = price / 10;

// PREPARE ALL INPUT VARIABLES
input[0] = global.type[global.simstep]; // BINARY
input[1] = global.volume[global.simstep]/100000000;
//input[2] = global.region[global.simstep]/52;
//input[2] = global.week[global.simstep]/52;
input[2] = global.t4046_percentage[global.simstep];
input[3] = global.t4225_percentage[global.simstep];
input[4] = global.t4770_percentage[global.simstep];
input[5] = global.small_bags_percentage[global.simstep];
input[6] = global.large_bags_percentage[global.simstep];
input[7] = global.xlarge_bags_percentage[global.simstep];

// CALL THE NETWORK AND GET THE OUTPUT
output = mlp.Forward(input);
predicted_price = output[0] * 3;

if (top = 1)
{
show_debug_message(string(price) + " " + string(predicted_price) + " = " + string(abs(price - predicted_price)));

/*
show_debug_message(input[0]);
show_debug_message(input[1]);
show_debug_message(input[2]);
show_debug_message(input[3]);
show_debug_message(input[4]);
*/
}

rating = rating + abs(price - predicted_price);

if (top = 1)
	{
	global.bestel_rating = string_format(rating / global.s,0,7);	
	}
}

// CREATE THE RESTART FUNCTION
Restart = function()
	{
	// SET THE RATING
	rating = 0;
	}

// SAVE THE GOOD VALUES
Restart();