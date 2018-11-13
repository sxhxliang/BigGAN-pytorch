def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
	total_parameters = 0
	parameters_string = ""

	for variable in tf.trainable_variables():

		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		if len(shape) == 1:
			parameters_string += ("%s %d, " % (variable.name, variable_parameters))
		else:
			parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

	if output_to_logging:
		if output_detail:
			logging.info(parameters_string)
		logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
	else:
		if output_detail:
			print(parameters_string)
		print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
	total_parameters = 0
	parameters_string = ""

	for variable in tf.all_variables():

		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		if len(shape) == 1:
			parameters_string += ("%s %d, " % (variable.name, variable_parameters))
		else:
			parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

	if output_to_logging:
		if output_detail:
			logging.info(parameters_string)
		logging.info("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))
	else:
		if output_detail:
			print(parameters_string)
		print("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))
print_num_of_total_parameters()