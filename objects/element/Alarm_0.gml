if (top = 1)
{
var fileName = working_directory + "/genetic.ai";
var buffer = buffer_load(fileName);
mlp.Load(buffer);
buffer_delete(buffer);

type = val1.value;
volume = val2.value;
t4046 = val3.value;
t4225 = val4.value;
t4770 = val5.value;
small_bags = val6.value;
large_bags = val7.value;
xlarge_bags = val8.value;

tsum = (t4046 + t4225 + t4770);
bagsum = (small_bags + large_bags + xlarge_bags);

t4046_percentage = t4046/tsum;
t4225_percentage = t4225/tsum;
t4770_percentage = t4770/tsum;

small_bags_percentage = small_bags/bagsum;
large_bags_percentage = large_bags/bagsum;
xlarge_bags_percentage = xlarge_bags/bagsum;

if (type = "organic") {type = 0;}
if (type = "conventional") {type = 1;}


v1 = real(type); // BINARY
v2 = real(volume/100000000);
v3 = real(t4046_percentage);
v4 = real(t4225_percentage);
v5 = real(t4770_percentage);
v6 = real(small_bags_percentage);
v7 = real(large_bags_percentage);
v8 = real(xlarge_bags_percentage);

// CALL THE NETWORK AND GET THE OUTPUT
output = mlp.Forward([v1,v2,v3,v4,v5,v6,v7,v8]);
predicted_price = output[0] * 3;

show_message_async("PREDICTED PRICE: " + string(abs(predicted_price)));
}