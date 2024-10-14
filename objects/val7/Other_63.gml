var i_d = ds_map_find_value(async_load, "id");
if i_d == val
{
    if ds_map_find_value(async_load, "status")
    {
        if ds_map_find_value(async_load, "result") != ""
        {
            value = ds_map_find_value(async_load, "result");
        }
    }
}