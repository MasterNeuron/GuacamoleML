var fileName = get_save_filename("ai", "genetic");
var buffer = genetic.bestel.mlp.Save();
var sizeOf = buffer_sizeof(NumberType.DOUBLE);
var bufferCopyTest = buffer_create(buffer_get_size(buffer), buffer_fixed, sizeOf);
buffer_copy(buffer, 0, buffer_get_size(buffer), bufferCopyTest, 0);
buffer_save(buffer, fileName);
buffer_delete(buffer);
buffer_delete(bufferCopyTest);
show_message_async("MLP SAVED!");