var fileName = get_open_filename("ai", "genetic");
var buffer = buffer_load(fileName);
genetic.bestel.mlp.Load(buffer);
buffer_delete(buffer);
show_message_async("MLP LOADED!");