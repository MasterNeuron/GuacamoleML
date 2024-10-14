var data = "";
data = data + "CURRENT POINT: " + string(global.simstep) + "\nALL POINTS: " + string(global.simstepend) + "\nCOUNT: " + string(global.totalsteps) + "\n";
data = data + "GENERATION: " + string(global.generation) + "\n";
data = data + "AVERAGE ERROR: " + string(global.bestel_rating) + "\n";

draw_set_font(ft_courier);
draw_set_valign(fa_top);
draw_set_halign(fa_left);
draw_set_color(c_black);
draw_text(10,10,data);