from typing import Tuple
from contracts.v1.preprocessing import Preprocessing
from Trainer import Trainer

class PreprocessingImpl(Preprocessing):
    def do_preprocessing_and_return_preprocessed_variables(self, user_inputs) -> Tuple[dict, str]:
        trainer = Trainer(user_inputs)
        trainer.train()

        preprocessed_data = {'x_train_l' : trainer.x_train_l, 'x_train_u' : trainer.x_train_u, 'x_test' : trainer.x_test, 
                             'labels_train' : trainer.class_labels_train, 'labels_test' : trainer.class_labels_test, 
                             'class_names' : trainer.class_names}
        
        return preprocessed_data, 'Preprocessing Done'
