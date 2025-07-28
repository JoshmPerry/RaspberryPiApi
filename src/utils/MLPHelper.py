def train_valid_split(raw_data, labels, split_index):
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]