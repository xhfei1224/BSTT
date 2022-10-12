import argparse
from DataProcess import *
from Evaluations import *
from MMCNN_model import *

def main():
	parser = argparse.ArgumentParser(description = 'input the dataset dir path.')

	parser.add_argument('--ISRUCpath',
					   type = str,
					   default = 'ISRUC/',
					   help = 'File path to the dataset ISRUC')
	parser.add_argument('--MASSpath',
					   type = str,
					   default = 'MASS/',
					   help = 'input the MASS dataset dir')
	parser.add_argument('--choosedata',
						type = str,
						default = 'ISRUC',
						help = 'choose ISRUC:input ISRUC,choose MASS:input MASS')
						
	args = parser.parse_args()

	
	data_2a_path = args.ISRUCpath
	data_2a_files = ["subject1","subject2","subject3","subject4","subject5"]
	data_2b_path = args.MASSpath
	data_2b_files = ["subject1","subject2","subject3","subject4","subject5"]
	if args.choosedata == 'ISRUC':
		choose2aor2b = 1
		choose2aclasses = 2
		GetData = DataProcess(data_2a_path,data_2a_files,choose2aor2b,choose2aclasses)		
	elif args.choosedata == 'MASS':
		choose2aor2b = 2
		GetData = DataProcess(data_2b_path,data_2b_files,choose2aor2b)
	else:
		GetData = DataProcess(data_2a_path,data_2a_files,1,2)
	data = GetData2a.data
	label = GetData2a.label

	k = 5
	window_long = 3000
	window_val_interval = 10
	window_test_interval = 50

	num_validation_samples = len(data)//k

	validation_scores = []# validation scores
	histories = []        # history
	matrixes = []         # Confusion matrix
	kappas = []           # kappa 
	f1_scores = []        # f1 scores
	loss_scores = []           # loss
	errors = []           # errors
	for fold in range(k):
		validation_data = data[num_validation_samples * fold:num_validation_samples * (fold+1)]
		validation_label = label[num_validation_samples * fold:num_validation_samples * (fold+1)]
		train_data = np.concatenate((data[:num_validation_samples * fold],data[num_validation_samples*(fold + 1):]),
									 axis=0)
		train_label = np.concatenate((label[:num_validation_samples * fold],label[num_validation_samples*(fold + 1):]),
									 axis=0)
		# shuffle
		index_validation = [i for i in range(len(validation_data))] 
		random.shuffle(index_validation)
		validation_data = validation_data[index_validation]
		validation_label = validation_label[index_validation]

		index_train = [i for i in range(len(train_data))] 
		random.shuffle(index_train)
		train_data = train_data[index_train]
		train_label = train_label[index_train]
		# data augmentation

		# Sliding windows
		data_aug_train,label_aug_train = GetData.data_augmentation(train_data,
														   train_label,
														   windows_long = window_long,
														   interval = window_val_interval)
		train_data = data_aug_train
		train_label = label_aug_train

		# Gauss
		data_aug_train_gauss,label_aug_train_gauss = GetData.gauss_data_augmentation(train_data,train_label,0.005,m = 2)
		train_data = data_aug_train_gauss
		train_label = label_aug_train_gauss

		# validation data cut
		data_aug_validation,label_aug_validation = GetData.data_augmentation(validation_data,
																	validation_label,
																	windows_long = window_long,
																	interval = window_test_interval)
		validation_data = data_aug_validation
		validation_label = label_aug_validation
		print(validation_data.shape,"\n",validation_label.shape)
		print(train_data.shape,"\n",train_label.shape)
		
		MMCNN = MMCNN_model(channels = 6,samples = 3000)
		model = MMCNN.model
		early_stopping = EarlyStopping(monitor='val_loss',patience=30,verbose=0,mode='auto')
		history = model.fit(train_data,
							train_label,
							epochs=50,
							batch_size=128,
							callbacks=[early_stopping],
							shuffle=True,
							validation_data = (validation_data,validation_label))
		loss_score,error,validation_score = model.evaluate(validation_data,validation_label)
		Result = Evaluations(history,
							  model.predict(validation_data),
							  validation_label,
							  loss_score,
							  error,
							  validation_score)
		
		histories.append(Result.history)
		matrixes.append(Result.matrix)
		kappas.append(Result.kappa)
		f1_scores.append(Result.f1)
		validation_scores.append(Result.validation_score)
		loss_scores.append(Result.loss_score)
		errors.append(Result.error)
	ave_val_score = np.average(validation_scores)
	ave_f1 = np.average(f1_scores)
	ave_kappa = np.average(kappas)
	ave_loss = np.average(loss_scores)
	ave_error = np.average(errors)

	print("the average validation score:",ave_val_score)
	print("the average f1 score:",ave_f1)
	print("the average kappa:",ave_kappa)
	print("the average loss:",ave_loss)
	print("the average error:",ave_error)

if __name__ == "__main__":
	main()