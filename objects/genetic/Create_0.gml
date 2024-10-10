var num = 0;
// OPEN THE DATA FILE
var file = file_text_open_read("/Users/inwhy/Desktop/avocado.csv");
while (!file_text_eof(file))
{
	var str = file_text_read_string(file);
    file_text_readln(file);
	
	// START FROM THE SECOND LINE
	if (num > 0)
	{
	data_array = string_split(str, ",");
	
	// global.idd[num] = data_array[0]; // ID
	global.date[num] = data_array[1]; // DATE
	global.price[num] = real(data_array[2]); // AVERAGE PRICE
	global.volume[num] = real(data_array[3]); // TOTAL VOLUME
	global.t4046[num] = real(data_array[4]); // 4046
	global.t4225[num] = real(data_array[5]); // 4225
	global.t4770[num] = real(data_array[6]); // 4770
	// global.total_bags[num] = data_array[7]; // TOTAL BAGS
	global.small_bags[num] = real(data_array[8]); // SMALL BAGS
	global.large_bags[num] = real(data_array[9]); // LARGE BAGS
	global.xlarge_bags[num] = real(data_array[10]); // XLARGE BAGS
	global.type[num] = data_array[11]; // TYPE
	global.year[num] = data_array[12]; // YEAR
	global.region[num] = data_array[13]; // REGION
	
	var total_plu = (global.t4046[num] + global.t4225[num] + global.t4770[num]);
	
	global.t4046_percentage[num] = global.t4046[num]/total_plu;
	global.t4225_percentage[num] = global.t4225[num]/total_plu;
	global.t4770_percentage[num] = global.t4770[num]/total_plu;
	
	var total_bags = (global.small_bags[num] + global.large_bags[num] + global.xlarge_bags[num]);
	
	global.small_bags_percentage[num] = global.small_bags[num]/total_bags;
	global.large_bags_percentage[num] = global.large_bags[num]/total_bags;
	global.xlarge_bags_percentage[num] = global.xlarge_bags[num]/total_bags;
	global.total_bags[num] = total_bags;
	
	date = string_split(global.date[num],"-");
	global.year[num] = real(date[0]);
	global.month[num] = real(date[1]);
	global.day[num] = real(date[2]);
	
	global.date_time[num] = date_create_datetime(global.year[num],global.month[num],global.day[num],1,1,1);
	global.week[num] = date_get_week(global.date_time[num]);
	
	if (global.type[num] = "organic") {global.type[num] = 0;}
	if (global.type[num] = "conventional") {global.type[num] = 1;}
	
if (global.region[num] = "Albany") {global.region[num] = 1;}
if (global.region[num] = "Atlanta") {global.region[num] = 2;}
if (global.region[num] = "BaltimoreWashington") {global.region[num] = 3;}
if (global.region[num] = "Boise") {global.region[num] = 4;}
if (global.region[num] = "Boston") {global.region[num] = 5;}
if (global.region[num] = "BuffaloRochester") {global.region[num] = 6;}
if (global.region[num] = "California") {global.region[num] = 7;}
if (global.region[num] = "Charlotte") {global.region[num] = 8;}
if (global.region[num] = "Chicago") {global.region[num] = 9;}
if (global.region[num] = "CincinnatiDayton") {global.region[num] = 10;}
if (global.region[num] = "Columbus") {global.region[num] = 11;}
if (global.region[num] = "DallasFtWorth") {global.region[num] = 12;}
if (global.region[num] = "Denver") {global.region[num] = 13;}
if (global.region[num] = "Detroit") {global.region[num] = 14;}
if (global.region[num] = "GrandRapids") {global.region[num] = 15;}
if (global.region[num] = "GreatLakes") {global.region[num] = 16;}
if (global.region[num] = "HarrisburgScranton") {global.region[num] = 17;}
if (global.region[num] = "HartfordSpringfield") {global.region[num] = 18;}
if (global.region[num] = "Houston") {global.region[num] = 19;}
if (global.region[num] = "Indianapolis") {global.region[num] = 20;}
if (global.region[num] = "Jacksonville") {global.region[num] = 21;}
if (global.region[num] = "LasVegas") {global.region[num] = 22;}
if (global.region[num] = "LosAngeles") {global.region[num] = 23;}
if (global.region[num] = "Louisville") {global.region[num] = 24;}
if (global.region[num] = "MiamiFtLauderdale") {global.region[num] = 25;}
if (global.region[num] = "Midsouth") {global.region[num] = 26;}
if (global.region[num] = "Nashville") {global.region[num] = 27;}
if (global.region[num] = "NewOrleansMobile") {global.region[num] = 28;}
if (global.region[num] = "NewYork") {global.region[num] = 29;}
if (global.region[num] = "Northeast") {global.region[num] = 30;}
if (global.region[num] = "NorthernNewEngland") {global.region[num] = 31;}
if (global.region[num] = "Orlando") {global.region[num] = 32;}
if (global.region[num] = "Philadelphia") {global.region[num] = 33;}
if (global.region[num] = "PhoenixTucson") {global.region[num] = 34;}
if (global.region[num] = "Pittsburgh") {global.region[num] = 35;}
if (global.region[num] = "Plains") {global.region[num] = 36;}
if (global.region[num] = "Portland") {global.region[num] = 37;}
if (global.region[num] = "RaleighGreensboro") {global.region[num] = 38;}
if (global.region[num] = "RichmondNorfolk") {global.region[num] = 39;}
if (global.region[num] = "Roanoke") {global.region[num] = 40;}
if (global.region[num] = "Sacramento") {global.region[num] = 41;}
if (global.region[num] = "SanDiego") {global.region[num] = 42;}
if (global.region[num] = "SanFrancisco") {global.region[num] = 43;}
if (global.region[num] = "Seattle") {global.region[num] = 44;}
if (global.region[num] = "SouthCarolina") {global.region[num] = 45;}
if (global.region[num] = "SouthCentral") {global.region[num] = 46;}
if (global.region[num] = "Southeast") {global.region[num] = 47;}
if (global.region[num] = "Spokane") {global.region[num] = 48;}
if (global.region[num] = "StLouis") {global.region[num] = 49;}
if (global.region[num] = "Syracuse") {global.region[num] = 50;}
if (global.region[num] = "Tampa") {global.region[num] = 51;}
if (global.region[num] = "TotalUS") {global.region[num] = 52;}
if (global.region[num] = "West") {global.region[num] = 53;}
if (global.region[num] = "WestTexNewMexico") {global.region[num] = 54;}

// Side 1: Northeast
if (
    global.region[num] == 1 ||  // Albany
    global.region[num] == 5 ||  // Boston
    global.region[num] == 6 ||  // BuffaloRochester
    global.region[num] == 17 || // HarrisburgScranton
    global.region[num] == 18 || // HartfordSpringfield
    global.region[num] == 29 || // NewYork
    global.region[num] == 30 || // Northeast
    global.region[num] == 31 || // NorthernNewEngland
    global.region[num] == 33 || // Philadelphia
    global.region[num] == 35 || // Pittsburgh
    global.region[num] == 50    // Syracuse
) {
    global.side[num] = 1;
}

// Side 2: Midwest
else if (
    global.region[num] == 9  || // Chicago
    global.region[num] == 10 || // CincinnatiDayton
    global.region[num] == 11 || // Columbus
    global.region[num] == 14 || // Detroit
    global.region[num] == 15 || // GrandRapids
    global.region[num] == 16 || // GreatLakes
    global.region[num] == 20 || // Indianapolis
    global.region[num] == 36 || // Plains
    global.region[num] == 49    // StLouis
) {
    global.side[num] = 2;
}

// Side 3: South
else if (
    global.region[num] == 2  || // Atlanta
    global.region[num] == 3  || // BaltimoreWashington
    global.region[num] == 8  || // Charlotte
    global.region[num] == 12 || // DallasFtWorth
    global.region[num] == 19 || // Houston
    global.region[num] == 21 || // Jacksonville
    global.region[num] == 24 || // Louisville
    global.region[num] == 25 || // MiamiFtLauderdale
    global.region[num] == 26 || // Midsouth
    global.region[num] == 27 || // Nashville
    global.region[num] == 28 || // NewOrleansMobile
    global.region[num] == 32 || // Orlando
    global.region[num] == 38 || // RaleighGreensboro
    global.region[num] == 39 || // RichmondNorfolk
    global.region[num] == 40 || // Roanoke
    global.region[num] == 45 || // SouthCarolina
    global.region[num] == 46 || // SouthCentral
    global.region[num] == 47 || // Southeast
    global.region[num] == 51 || // Tampa
    global.region[num] == 54    // WestTexNewMexico
) {
    global.side[num] = 3;
}

// Side 4: West
else if (
    global.region[num] == 4  || // Boise
    global.region[num] == 7  || // California
    global.region[num] == 13 || // Denver
    global.region[num] == 22 || // LasVegas
    global.region[num] == 23 || // LosAngeles
    global.region[num] == 34 || // PhoenixTucson
    global.region[num] == 37 || // Portland
    global.region[num] == 41 || // Sacramento
    global.region[num] == 42 || // SanDiego
    global.region[num] == 43 || // SanFrancisco
    global.region[num] == 44 || // Seattle
    global.region[num] == 48 || // Spokane
    global.region[num] == 53    // West
) {
    global.side[num] = 4;
}

/*
	show_debug_message(global.date[num]);
	show_debug_message(global.price[num]);
	show_debug_message(global.volume[num]);
	show_debug_message(global.t4046[num]);
	show_debug_message(global.t4225[num]);
	show_debug_message(global.t4770[num]);
	show_debug_message(global.small_bags[num]);
	show_debug_message(global.large_bags[num]);
	show_debug_message(global.xlarge_bags[num]);
	show_debug_message(global.type[num]);
	show_debug_message(global.year[num]);
	show_debug_message(global.region[num]);
	show_debug_message(" ");
*/	

	}
	
	num++;
}
file_text_close(file);

global.generation = 1;
global.simstep = 1;
global.simstepend = num - 1;
global.bestel_rating = 0;
global.maxsteps = 120;
global.totalsteps = global.maxsteps;
global.s = 0;

// CREATE THE ELEMENTS ARRAY
elements = [];
elbrains = [];

// ELEMENTS COUNT
var num = 300;
repeat(num)
	{
	// GENETIC ELEMENT CREATION
	el = instance_create_depth(0,0,0,element);
	// PUSH INTO ARRAY
	array_push(elements, el);
	}
// SET SOME RANDOM NN TO BE THE TOP ONE
bestel = el;
bestel.top = 1;

// DROP THE BEAT
alarm[0] = 1;