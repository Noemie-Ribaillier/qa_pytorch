#####################################################################################################################
#####                                                                                                           #####
#####                    TRANSFORMER NETWORK APPLICATION: QUESTION ANSWERING (USING PYTORCH)                    #####
#####                                           Created on: 2025-03-23                                          #####
#####                                                                                                           #####
#####################################################################################################################

#####################################################################################################################
#####                                                  PACKAGES                                                 #####
#####################################################################################################################

# Clear the environment
globals().clear()

# Load the libraries
import numpy as np
import random
import re
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast
from sklearn.metrics import f1_score
from transformers import DistilBertForQuestionAnswering
from transformers import Trainer
from transformers import TrainingArguments
import torch

# Set the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/qa_pytorch')

# Set the random seed for reproducibility
seed = 21
# Set the random seed for Python's random module
random.seed(seed)
# Set the random seed for Python's numpy module
np.random.seed(seed)
# Set the random seed for Python's torch module
torch.manual_seed(seed)
# To ensure deterministic behavior
torch.backends.cudnn.deterministic = True


#####################################################################################################################
#####                                   DATA LOADING, CLEANING & PREPROCESSING                                  #####
#####################################################################################################################

# Load the dataset 
babi_dataset = load_from_disk('data/')

# Determine and print a random example of the training set to have a first look
rd_index = random.randint(0,len(babi_dataset['train'])-1)
print(babi_dataset['train'][rd_index])

# Control that the entire dataset of stories has the same format (first the 2 context then the question so type = [0, 0, 1])
# Create an empty set (to gather all different types of the dataset)
type_set = set()
# Iterate on items from the train dataset
for story in babi_dataset['train']:
    # Add all the new types
    if str(story['story']['type'] ) not in type_set:
        type_set.add(str(story['story']['type'] ))
print(type_set)

# Do the same for supporting_ids (to check that they all have the same format because we use it later)
supporting_ids_set = set()
for story in babi_dataset['train']:
    if str(story['story']['supporting_ids'] ) not in supporting_ids_set:
        supporting_ids_set.add(str(story['story']['supporting_ids'] ))
print(supporting_ids_set)

# Flatten the dataset: transform from nested dictionnary to just 1 dictionnary for each train/test set
flattened_babi = babi_dataset.flatten()

# Print the dimensions of train and test sets (dictionnary)
len(flattened_babi['train']) ; len(flattened_babi['test'])

# Print a random example of the training set to have a first look
flattened_babi['train'][rd_index]

# Create the function to extract the information (answer, question, and context) from the story, and join the context into a single entry
def extract_information(story):
    '''
    Create the function to extract the useful information (question, context and answer) from the story dictionnary

    Inputs: 
    story -- input dictionnary containing more information than we need

    Returns:
    output_dict -- output dictionnary containing only the information we need (question, context and answer)
    '''
    # Create an empty dictionnary
    output_dict = {}

    # Extract the question
    output_dict['question'] = story['story.text'][2]

    # Extract and join the context (we use hard code index because we checked previously that all the data has the same format)
    output_dict['context'] = ' '.join([story['story.text'][0], story['story.text'][1]])
    
    # Extract the answer
    output_dict['answer'] = story['story.answer'][2]

    return output_dict


# Process the data to extract the information we need (context, question, answer), for train and test sets
processed_babi = flattened_babi.map(extract_information)

# Print a random example of the training and test sets to have a first look
processed_babi['train'][rd_index]
processed_babi['test'][rd_index]


# Create the function to get the start and end indexes of the answer in each of the stories in the dataset
def get_start_end_idx(story):
    '''
    Get the start and end indexes of the answer in each of the stories in the dataset

    Inputs:
    story -- input dictionnary containing information about the context, question and answer

    Returns:
    idx_dict -- output dictionnary containing information about the start and end of the answer
    '''
    # Get the start index of the answer
    # Get all indexes, because we noticed the answer word can appear in both sentences making the context
    str_idxs = [match.start() for match in re.finditer(story['answer'], story['context'])]
    # Get the index (corresponding to the context index), if only one, take it as it is
    if len(str_idxs)==1:
        str_idx = str_idxs[0]
    # Get the index (corresponding to the context index), if there are several, take the one corresponding to the right context (-1 because it doesn't start at 0)
    else:
        str_idx = str_idxs[int(story['story.supporting_ids'][2][0])-1]

    # Get the end index (meaning the index of the next character [after the last character of the answer])
    end_idx = str_idx + len(story['answer'])

    # Create the index dictionnary (containing the start and end index of the answer)
    idx_dict = {'str_idx':str_idx, 'end_idx': end_idx}

    return idx_dict


# Process the data to get the start and end index
processed_babi = processed_babi.map(get_start_end_idx)

# Print a random example of the training set to have a first look
processed_babi['train'][rd_index]


#####################################################################################################################
#####                            TOKENIZE AND ALIGN LABELS WITH HUGGING FACE LIBRARY                            #####
#####################################################################################################################

# Load the pre-trained tokenizer (specific to the model we are going to use)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create the function to align the start and end indices with the tokens associated with the target answer word
def tokenize_align(story):
    '''
    Align the start and end indices with the token(s) associated with the target answer word
    Indeed, when tokenizing, words can be split into subwords, 
    which can create some misalignment between the list of tags for the dataset and the list of labels generated by the tokenizer
    
    Inputs:
    story -- input (dict)

    Returns:
    tokenized_aligned_dict -- the output dictionnary containing the input_ids, the attention_mask, the start_positions and the end_positions
    '''
    # Tokenize context and question from the input (transform from words/subworks to index)
    encoding = tokenizer(story['context'], story['question'], truncation=True, padding=True, max_length=tokenizer.model_max_length)
    
    # Define the start and end positions
    start_positions = encoding.char_to_token(story['str_idx'])
    # Use -1 because end_idx is exclusive (take the next character, after the answer)
    end_positions = encoding.char_to_token(story['end_idx']-1)

    # If start_positions is not found, default goes to max encoding
    if start_positions is None:
        start_positions = tokenizer.model_max_length
    # If end_positions is not found, default goes to max encoding
    if end_positions is None:
        end_positions = tokenizer.model_max_length
    
    # Create the output dictionnary with the following 4 elements: input_ids, attention_mask, start_positions and end_positions
    tokenized_aligned_dict = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'],
                              'start_positions': start_positions, 'end_positions': end_positions}
    
    return tokenized_aligned_dict


# Map the processed_babi data to the tokenize_align function
qa_dataset = processed_babi.map(tokenize_align)

# Get all column names
qa_dataset["train"].column_names

# Remove the columns we don't need
qa_dataset = qa_dataset.remove_columns(['story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'])

# Print a random example of the training set to have a first look
qa_dataset['train'][rd_index]


#####################################################################################################################
#####                                              TRAIN THE MODEL                                              #####
#####################################################################################################################

# Split qa_dataset into train and test datasets
train_ds = qa_dataset['train']
test_ds = qa_dataset['test']

# Define the columns we want to keep
columns_to_keep = ['input_ids','attention_mask', 'start_positions', 'end_positions']

# Set the right format (PyTorch Tensor, format compatible with ML models, particularly those from the HF Transformers library) for train and test datasets
train_ds.set_format(type='pt', columns=columns_to_keep)
test_ds.set_format(type='pt', columns=columns_to_keep)


# Create the function to compute the F1 score for start and end index (using macro method)
def f1_metric(pred):
    '''
    Compute the F1 score for start index and end index (using macro method)

    Inputs:
    pred -- array containing the true start/end indexes and their predictions

    Returns:
    f1_dict -- dictionnary containing the f1 score for start index and end index
    '''
    # Extract answer positions (start and end, true and predicted values)
    # Get the true start index (1st element)
    start_labels = pred.label_ids[0]
    # Apply the max to the last axis to get the predicted index (among the 1st elements to get start index)
    start_preds = pred.predictions[0].argmax(-1)
    # Get the true end index (2nd element)
    end_labels = pred.label_ids[1]
    # Apply the max to the last axis to get the predicted index (among the 2nd elements to get end index)
    end_preds = pred.predictions[1].argmax(-1)
    
    # Use the F1 score
    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')
    
    # Create the dictionnary with the F1 score for start index and for end index
    f1_dict = {'f1_start': f1_start, 'f1_end': f1_end,}
    
    return f1_dict


# Load the pre-trained model
pytorch_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Set up the training arguments
training_args = TrainingArguments(
    # Output (where the model checkpoints and final model will be saved)
    output_dir = 'results',
    # Whether to overwrite the output directory
    overwrite_output_dir = True,
    # Number of training epoches
    num_train_epochs = 10,
    # Batch size per device during training (number of samples that will be passed through the model in one step)
    per_device_train_batch_size = 8,
    # Batch size for evaluation
    per_device_eval_batch_size = 8,
    # Number of steps to perform learning rate warmup
    warmup_steps = 20,
    # Control the strength of weight decay (regularization)
    weight_decay = 0.01,
    # Directory where logs will be saved
    logging_dir=None,
    # Specify how often to log training information
    logging_steps=50
)

# Set up the trainer
trainer = Trainer(
    # Transformers model to be trained
    model = pytorch_model,
    # Training arguments, defined above
    args = training_args,
    # Training dataset
    train_dataset = train_ds,
    # Evaluation dataset
    eval_dataset = test_ds,
    # Metrics to use
    compute_metrics = f1_metric
)

# Train the model
trainer.train()


#####################################################################################################################
#####                                     EVALUATE THE MODEL ON TEST DATASET                                    #####
#####################################################################################################################

# Evaluate the model on test dataset
trainer.evaluate(test_ds)


#####################################################################################################################
#####                                     EVALUATE THE MODEL ON OUR OWN DATA                                    #####
#####################################################################################################################

# Check if we have a GPU or not (so CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Send the model to the device (CPU in our case)
pytorch_model.to(device)

# Determine the question and context 
question, context = 'What is west of the bedroom?','The kitchen is east of the bedroom. The garden is west of the bedroom.'

# Tokenize inputs
input_dict = tokenizer(context, question, return_tensors='pt')

# Send the inputs to the device (CPU in our case)
input_ids = input_dict['input_ids'].to(device)
attention_mask = input_dict['attention_mask'].to(device)

# Get the model predictions
outputs = pytorch_model(input_ids, attention_mask = attention_mask)

# Get the start and end logits
start_logits = outputs[0]
end_logits = outputs[1]

# Transform the ids (int) back to tokens (words/subwords)
all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])

# Extract answer positions (start and end) (+1 because it's exclusive)
answer = ' '.join(all_tokens[torch.argmax(start_logits, 1)[0] : torch.argmax(end_logits, 1)[0]+1])

# Print the question and answer
print(question, answer.capitalize())
