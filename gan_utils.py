
from transformers import BartTokenizer, BartModel,BartForConditionalGeneration
from torch.distributions import Categorical
import pandas as pd
import torch.nn as nn
from undecorated import undecorated
import numpy as np
from datasets import load_metric
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq

import torch.optim as optim
import torchvision.utils as vutils
import types
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import random


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


########## GAN models ###############



class Generator(nn.Module):
    def __init__(self,checkpoint='facebook/bart-base',vocab_size=50265): 
        super(Generator,self).__init__() 


        #Load Model with given checkpoint and extract its body
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)

        self.softmax = nn.Softmax()

    def forward(self,inputs):
        #forward pass which creates batch of outputs based on input
       # print(inputs)
        logprobs = []

        bsize = inputs['input_ids'].shape[0]
        output_tokens = torch.tensor([0]*bsize).type(torch.LongTensor).cuda()

        output_tokens = output_tokens.unsqueeze(-1)
        eos = 2
        done = False
        first = True
        unfinished_sequences = inputs['input_ids'].new(inputs['input_ids'].shape[0]).fill_(1)


        encoder_hidden_state = self.model.get_encoder()(inputs['input_ids'],attention_mask=inputs['attention_mask']).last_hidden_state
        while not done:
           # print(inputs['attention_mask'].shape)

           # print('output tokens')
           # print(output_tokens.shape)
            last_hidden = self.model.get_decoder()(input_ids=output_tokens,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=inputs['attention_mask'])
           # print(last_hidden.last_hidden_state - last_hidden[0])
            lm_logits = self.model.lm_head(last_hidden[0])

            lm_logits = (lm_logits + self.model.final_logits_bias)#.squeeze()

            lm_logits = lm_logits.squeeze()

            if not first:
                lm_logits = lm_logits[:,-1,:]
            first = False



           # lm_logits = nn.functional.batch_norm(lm_logits)
            cat = Categorical(logits=lm_logits)




           # try:
            next_word = cat.sample()#lm_logits.argmax(axis=1)#.unsqueeze(-1)
            
            valid = False


            next_word = next_word * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences.mul((next_word != tokenizer.eos_token_id).long())

            logprob = cat.log_prob(next_word)


            logprobs.append(logprob.unsqueeze(1))


            output_tokens = torch.cat([output_tokens,torch.tensor(next_word).unsqueeze(1)],axis=1)
            #print(output_tokens)
            valid = True
            for j in range(output_tokens.shape[0]):
                if not output_tokens[j].__contains__(eos):
                    valid = False
            if valid or output_tokens.shape[1] >=16:
                break
        logprobs= torch.cat(logprobs,axis=1)
       
        return output_tokens[:,1:],logprobs


class Discriminator(nn.Module):
    def __init__(self,num_labels=1,checkpoint='BART-base/checkpoint-25000'): 
        super(Discriminator,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint).model.encoder
        self.dropout = nn.Dropout(0.10) #idk if i should include dropout ... maybe bc these models are prone to mode collapse??
        self.classifier = nn.Linear(768,num_labels) # load and initialize weights
       # self.logit = torch.logit()

    def forward(self, input_ids=None,attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        mean_embedding = torch.mean(sequence_output[:,:,:],axis=1)

       # print(mean_embedding.shape)

       # print(sequence_output[:,-1,:].shape)

        logits = (self.classifier(mean_embedding) )# calculate losses


        return logits

class Discriminator_Titles(nn.Module):
    def __init__(self,num_labels=1,checkpoint='facebook/bart-base'): 
        super(Discriminator_Titles,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint).model.encoder
        self.dropout = nn.Dropout(0.10) #idk if i should include dropout ... maybe bc these models are prone to mode collapse??
        self.classifier = nn.Linear(768*2,num_labels) # load and initialize weights
       # self.logit = torch.logit()

    def forward(self, input_ids=None, title_embedding=None,attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        mean_embedding = torch.mean(sequence_output[:,:,:],axis=1)

       # print(mean_embedding.shape)
      #  print(title_embedding_mean.shape)
       # print(sequence_output[:,-1,:].shape)
        class_in = torch.cat([title_embedding,mean_embedding],axis=1)
        #print(class_in.shape)
        logits = (self.classifier(class_in) )# calculate losses


        return logits






#################### GAN training Functions ##################################


def discriminator_train_standard(discriminator,samples,labels,criterion,mask=None):
    '''
    This function trains the discriminator based on a batch of samples and labels
    We now train by asking the discriminator to predict if each prefix is valid, not just the entire scentence
    '''
    full_len = samples.shape[1]
    #print(full_len)
    outputs = []


    for i in range(0,full_len):
        
        curr_samples = samples[:,:i+1]


        pred = discriminator(curr_samples)[:,-1]#.view(-1)

        #outputs.append(pred.view(-1))

       # a = 1/0
        #if all have ended, stop NOTE this is a very inefficient implementation change it
        #shouldnt bother discriminating if the poem has already ended
        if mask != None:
           
            outputs.append((pred* mask[:,i]))
            if i == 0:
                
                errD = criterion(pred, labels)* mask[:,i]
            else:
                
                errD += criterion(pred, labels)* mask[:,i]
        else:
            outputs.append(pred.view(-1))
            if i == 0:
                errD = criterion(pred, labels)
            else:
                errD += criterion(pred, labels)
            #errD = err * mask[i]
        valid = True
        for j in range(samples.shape[0]):
            if not curr_samples[j].__contains__(2):
                valid = False
        if valid:
            break
   # print('out 0')
   # print(outputs[0])
   # print(len(outputs))
   # print(len(outputs[0]))
   # print(errD)
   # a = 1/0
    out = torch.stack(outputs).T
   # print(errD)
    return out,torch.mean(errD) #.detach()
    



def discriminator_train_title(discriminator,samples,labels,titles,criterion,mask=None):
    '''
    This function trains the discriminator based on a batch of samples and labels
    We now train by asking the discriminator to predict if each prefix is valid, not just the entire scentence
    '''
    full_len = samples.shape[1]
    #print(full_len)
    outputs = []

    title_embedding = torch.mean(discriminator.model(titles).last_hidden_state,axis=1)#[:,-1,:]
   # print(title_embedding)
    for i in range(0,full_len):
        
        curr_samples = samples[:,:i+1]


        pred = discriminator(curr_samples,title_embedding)[:,-1]#.view(-1)


        if mask != None:
           
            outputs.append((pred* mask[:,i]))
            if i == 0:
                
                errD = criterion(pred, labels)* mask[:,i]
            else:
                
                errD += criterion(pred, labels)* mask[:,i]
        else:
            outputs.append(pred.view(-1))
            if i == 0:
                errD = criterion(pred, labels)
            else:
                errD += criterion(pred, labels)
            #errD = err * mask[i]
        valid = True
        for j in range(samples.shape[0]):
            if not curr_samples[j].__contains__(2):
                valid = False
        if valid:
            break

    out = torch.stack(outputs).T
   # print(errD)
    return out,torch.mean(errD) #.detach()



def reinforce_loss(disc_logits, gen_logprobs, gamma, decay, epoch, mask,end_mask,ewma_reward):
    '''
      The REINFORCE loss.
      Args:
          disc_logits: float tensor, shape [batch_size, sequence_length].
          gen_logprobs: float32 tensor, shape [batch_size, sequence_length]
          gamma: a float, discount factor for cumulative reward.
          decay: a float, decay rate for the EWMA baseline of REINFORCE.
      Returns:
        Float tensor, shape [batch_size, sequence_length], the REINFORCE loss for
        each timestep.
    '''
      # Assume 1 logit for each timestep.
    #disc_logits = disc_logits.T
    batch_size, sequence_length = disc_logits.shape
    disc_pred = nn.functional.sigmoid(disc_logits)
    
    MIN_SEQ_LEN = 6 #length at which no length punishment 
    #decays as 1- len/MIN_SEQ_LEN
   # print(disc_pred)
  # MaskGAN uses log(D), but this is more stable empirically.
    rewards = 2.0 * disc_pred - 1
   # print('disc_logits')
   # print(disc_logits)
  # Compute cumulative rewards.

   # print(rewards_list)
   # print(len(rewards_list))
    cumulative_rewards = []
    cumulative_rewards = torch.zeros((batch_size,sequence_length)).cuda()
    
    #negative decaying reward for 
    length_punishment = torch.zeros((batch_size,sequence_length)).cuda()
   # end_mask[0,1] = 0
    for t in range(0,sequence_length):
        if t < MIN_SEQ_LEN:
            punished_tensor = torch.ones((batch_size,)).cuda() * -(1 - t/MIN_SEQ_LEN)
            length_punishment[:,t] = torch.where(end_mask[:,t] != 1,length_punishment[:,t],punished_tensor)
        else:
            ones = torch.ones((batch_size,)).cuda()
            #if end after, its good increase reward
            length_punishment[:,t] = torch.where(end_mask[:,t] != 1,length_punishment[:,t],ones/6) 
       # print('cum_value')
       # print(cum_value.shape)
        for s in range(t, sequence_length):
           # print('rewards shape')
           # print(rewards[:,s].shape)
            cumulative_rewards[:,s] += torch.tensor(np.power(gamma, (s - t))).cuda() * rewards[:,s]
        #cumulative_rewards.append(cum_value)
    #cumulative_rewards = torch.stack(cumulative_rewards, axis=1)

    #cumulative_rewards.shape.assert_is_compatible_with([batch_size, sequence_length])

    #with tf.variable_scope("reinforce", reuse=tf.AUTO_REUSE):
    cumulative_rewards = cumulative_rewards + length_punishment

   # print('cumulative_rewards')
   # print(cumulative_rewards)
   # cumulative_rewards = cumulative_rewards[:,1:]
    mean_reward = torch.mean(cumulative_rewards)
   # print('mean')
   # print(mean_reward)
    ewma_reward = decay * ewma_reward + (1.0 - decay) * mean_reward
    #update_op = tf.assign(ewma_reward, new_ewma_reward)
    #ewma_reward = 

  # REINFORCE

    advantage = cumulative_rewards - ewma_reward#/10

    #this should encourage the model to not collapse to mode
    #l2_norm = nn.functional.relu(torch.sqrt(torch.sum(torch.exp(gen_logprobs)**2)) * .1 - .6)
    
    
    reg_2 = nn.functional.relu(torch.exp(gen_logprobs[:,1:]) - .995)*100 #the later values, would be close to 1 if repeating
   # print(torch.tensor([[0,0]]*batch_size).shape)
   # print(reg_2.shape)
    reg_2 = torch.cat([torch.tensor([[0]]*batch_size).cuda(),reg_2],axis=1)
   # print(reg_2)
    
    reg_2 = reg_2 * (1/(epoch +3)) #decay


    loss = -advantage.detach() * gen_logprobs +  reg_2 
   # print('l shape')

    loss = loss * mask

    return loss, cumulative_rewards, ewma_reward



def reinforce_loss_syllables(disc_logits, gen_logprobs, gamma, decay, epoch, mask,end_mask,syllables,tokens,ewma_reward,syllable_dict):
    '''
      The REINFORCE loss.
      Args:
          disc_logits: float tensor, shape [batch_size, sequence_length].
          gen_logprobs: float32 tensor, shape [batch_size, sequence_length]
          gamma: a float, discount factor for cumulative reward.
          decay: a float, decay rate for the EWMA baseline of REINFORCE.
          epoch: epoch
          mask: binary mask of pad tokens
          end_mask: binary mask of if there is an end token
          syllables: list of lists of syllables per line in desired poems
          tokens: tokens which were generated (used for syllable rewards)
          ewma_reward: moving average used to calcualte advantage
          syllable_dict: dictionary holding syllable value of each token
      Returns:
        Float tensor, shape [batch_size, sequence_length], the REINFORCE loss for
        each timestep.
    '''
      # Assume 1 logit for each timestep.
    #disc_logits = disc_logits.T
    batch_size, sequence_length = disc_logits.shape
    disc_pred = nn.functional.sigmoid(disc_logits)
    
    MIN_SEQ_LEN = 6 #length at which no length punishment 
    #decays as 1- len/MIN_SEQ_LEN
   # print(disc_pred)
  # MaskGAN uses log(D), but this is more stable empirically.
    rewards = 2.0 * disc_pred - 1
   # print('disc_logits')
   # print(disc_logits)
  # Compute cumulative rewards.

   # print(rewards_list)
   # print(len(rewards_list))
    cumulative_rewards = []
    cumulative_rewards = torch.zeros((batch_size,sequence_length)).cuda()
    
    #negative decaying reward for 
    length_punishment = torch.zeros((batch_size,sequence_length)).cuda()
    
    running_syllables = torch.zeros((batch_size,),dtype=torch.int32).cuda() #syllables since start token or new line 
    curr_line = torch.zeros((batch_size,),dtype=torch.int32).cuda() # current line (based on new lines) capped at 2
    desired_syllables = torch.zeros((batch_size,),dtype=torch.int32).cuda() #syllables curr line should have
   # end_mask[0,1] = 0
    for t in range(0,sequence_length):
        zeros = torch.zeros((batch_size,)).cuda()
        int_zeros = torch.zeros((batch_size,),dtype=torch.int32).cuda()
        ones = torch.ones((batch_size,)).cuda()
        int_ones = torch.ones((batch_size,),dtype=torch.int32).cuda()
        
        if t < MIN_SEQ_LEN:
            punished_tensor = torch.ones((batch_size,)).cuda() * -(1 - t/MIN_SEQ_LEN)
            length_punishment[:,t] = torch.where(end_mask[:,t] != 1,length_punishment[:,t],punished_tensor)
        else:
            
            #if end after, its good increase reward
            length_punishment[:,t] = torch.where(end_mask[:,t] != 1,length_punishment[:,t],ones/6) 
        
        
        #calculate current line 
        new_line = torch.where(tokens[:,t] == 50118,ones,zeros).int() #one if is new line

        curr_line += new_line 
        curr_line = torch.where(curr_line <= 2, curr_line, ones.int()*2)
        
        desired_syllables = torch.tensor([syllables[i][line] for i,line in enumerate(curr_line.tolist())]).cuda()

         
        
        running_syllables += torch.tensor([syllable_dict[tok] for tok in tokens[:,t].tolist()]).cuda()

        
        
        syl_diff = torch.tensor(desired_syllables - running_syllables,dtype=torch.int32).cuda()

        few_syllables_mask = torch.where(syl_diff > 0, int_zeros, int_ones) #mask of where fewer syllables than wanted 
        #(no reward yet)
        syl_diff = torch.where(syl_diff == 0,int_ones*10,syl_diff).float() #correct equivilent to -10 wrong syllables
        syl_diff = torch.mul(syl_diff,few_syllables_mask) * 1/10 #lambda, chosen so effect of reward is failry small

        rewards[:,t] += syl_diff.cuda()
        
        running_syllables = torch.where(tokens[:,t] != 50118,running_syllables,int_zeros) #if is new line, zero
        running_syllables = torch.where(tokens[:,t] != 0,running_syllables,int_zeros) #if is start, zero
        running_syllables = torch.where(tokens[:,t] != 2,running_syllables,int_zeros) #if is end, zero
        #add reward if syllables match, do not punish bc worried about 
       # print('cum_value')
       # print(cum_value.shape)
        for s in range(t, sequence_length):
           # print('rewards shape')
           # print(rewards[:,s].shape)
            cumulative_rewards[:,s] += torch.tensor(np.power(gamma, (s - t))).cuda() * rewards[:,s]
        #cumulative_rewards.append(cum_value)
    #cumulative_rewards = torch.stack(cumulative_rewards, axis=1)

    #cumulative_rewards.shape.assert_is_compatible_with([batch_size, sequence_length])

    #with tf.variable_scope("reinforce", reuse=tf.AUTO_REUSE):
    cumulative_rewards = cumulative_rewards + length_punishment

   # print('cumulative_rewards')
   # print(cumulative_rewards)
   # cumulative_rewards = cumulative_rewards[:,1:]
    mean_reward = torch.mean(cumulative_rewards)
   # print('mean')
   # print(mean_reward)
    ewma_reward = decay * ewma_reward + (1.0 - decay) * mean_reward
    #update_op = tf.assign(ewma_reward, new_ewma_reward)
    #ewma_reward = 

  # REINFORCE

    advantage = cumulative_rewards - ewma_reward#/10

  
    #this should encourage the model to not collapse to mode
    reg_2 = nn.functional.relu(torch.exp(gen_logprobs[:,1:]) - .995)*100 #the later values, would be close to 1 if repeating
   # print(torch.tensor([[0,0]]*batch_size).shape)
   # print(reg_2.shape)
    reg_2 = torch.cat([torch.tensor([[0]]*batch_size).cuda(),reg_2],axis=1)
   # print(reg_2)
    
    reg_2 = reg_2 * (1/(epoch +3)) #decay

    #length_punishment = length_punishment * gen_logprobs
    #len_reg = 

    loss = -advantage.detach() * gen_logprobs +  reg_2 #note orig had a - infront of advantage...
   # print('l shape')

    loss = loss * mask

    return loss, cumulative_rewards, ewma_reward



################### Latent Space Sampling ####################




def generate_random_input(bsize,device):
    ret_dict = {'input_ids':[],'attention_mask':[]}
    #ret_vectors = []
    for b in range(bsize):
        length = np.random.randint(2,6,)
        attention_mask = [1]
        rand_input = [0]
        for i in range(length):
            rand_input.append(np.random.randint(5,50260))
            attention_mask.append(1)
        rand_input.append(2)
        attention_mask.append(1)
        while len(rand_input) < 7:
            rand_input.append(1)
            attention_mask.append(0)
        ret_dict['input_ids'].append(rand_input)
        ret_dict['attention_mask'].append(attention_mask)
        #rand_input = torch.tensor([rand_input])
        #print(rand_input)
    ret_dict['input_ids'] = torch.tensor(ret_dict['input_ids']).to(device)
    ret_dict['attention_mask'] = torch.tensor(ret_dict['attention_mask']).to(device)

    return ret_dict


syllable_inputs = [tokenizer('[5,7,5];'),tokenizer('[6,7,5];'),
                   tokenizer('[5,7,6];'),tokenizer('[5,8,5];'),
                   tokenizer('[5,7,5];'),tokenizer('[6,8,5];'),
                   tokenizer('[5,8,6];'),tokenizer('[6,8,6];')]
#cut off  end char
for i in range(len(syllable_inputs)):
    syllable_inputs[i]['input_ids'] = syllable_inputs[i]['input_ids'][:-1]
    syllable_inputs[i]['attention_mask'] = syllable_inputs[i]['attention_mask'][:-1]


syllables_raw = [[5,7,5],[6,7,5],[5,7,6],[5,8,5],[5,7,5],[6,8,5],[5,8,6],[6,8,6]]

def generate_random_input_syllables(bsize,device):
    ret_dict = {'input_ids':[],'attention_mask':[]}
    #ret_vectors = []
    raw_syllables = []
    for b in range(bsize):
        length = np.random.randint(2,6,)
        
        syllables = np.random.randint(0,8)
        attention_mask = syllable_inputs[syllables]['attention_mask'].copy()
        rand_input = syllable_inputs[syllables]['input_ids'].copy()
        raw_syllables.append(syllables_raw[syllables])
       # print(rand_input)
        for i in range(length):
            rand_input.append(np.random.randint(5,50260))
            attention_mask.append(1)
        rand_input.append(2)
        attention_mask.append(1)
        while len(rand_input) < 16:
            rand_input.append(1)
            attention_mask.append(0)
        ret_dict['input_ids'].append(rand_input)
        ret_dict['attention_mask'].append(attention_mask)
        #rand_input = torch.tensor([rand_input])
        #print(rand_input)
    ret_dict['input_ids'] = torch.tensor(ret_dict['input_ids']).to(device)
    ret_dict['attention_mask'] = torch.tensor(ret_dict['attention_mask']).to(device)

    return ret_dict,raw_syllables




############## Masking util functions (for loss functions) ##################



def get_valid_mask(seq):
    #gets mask of non pad tokens 
   
    pad = tokenizer.pad_token_id

    ones = torch.ones((seq.shape)).cuda().type(torch.int64)
    zeroes = torch.zeros((seq.shape)).cuda().type(torch.int64)

    mask = torch.where(seq == pad, seq, ones)

    mask = torch.where(seq != pad, mask, zeroes)
  
    return mask




def get_end_mask(seq):
    #gets mask of end token
   
    end = 2#tokenizer.pad_token_id

    ones = torch.ones((seq.shape)).cuda().type(torch.int64)
    zeroes = torch.zeros((seq.shape)).cuda().type(torch.int64)

    mask = torch.where(seq != end, seq, ones)

    mask = torch.where(seq == end, mask, zeroes)

    return mask