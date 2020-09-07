"""
Developer:          Divya sasidharan 
Version:            v1.0 (released)
Date:               17.08.2020
             
Description:        Start of HMM model training and decoding.   
Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Initial draft version
1.02                                Updated the calls for zero and second order
1.03                                Updated the calls for zero and second order
1.04                                Updated the code for input configuration files without words
1.05                                Updated the code for cross corpus training and testing
"""
import os
import time 
import pickle
import HMM_Training_v1 as train
import HMM_Viterbi_zero as decode_0
import HMM_Viterbi_first as decode_1 
#import hmm_viterbi_zero_v1 as decode_0
#import HMM_Viterbi_first_v2 as decode_1 # for last 3 conf
import HMM_Viterbi_second as decode_2


def open_file(configNo,corpus_name,dataset):
    """Read the corpus file based on the input corpus_name.    
    Input:-corpus name [Hint: penn or genia or conll]
    Output:- corpus file content
    """
    fileLoc="drf/"+configNo+"/"
    filelist=os.listdir(fileLoc) # returns all the pickle files in drf folder    
    for file in filelist:        
        corpus = file.split('_')[1].split('.')[0] 
        set= file.split('_')[2].split('.')[0] 
        if corpus==corpus_name and set==dataset: 
            corpus_file=pickle.load(open(fileLoc+file,"rb"))
    return corpus_file
            
         
        
        
def main(configNo,tagger,corpus_name,corpus_name_test):
    """The main function for HMM Model."""
    print("Start of HMM_Main")  
    
    tagger=tagger
    corpus_name=corpus_name
    start_time = time.time()
    if corpus_name=="penn" or corpus_name=="conll" or corpus_name=="genia":        
        if int(tagger)==0:
            print("Start of training ")  
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset)             
            tagCount_out=train.tagCount(corpus)            
            transitionProbability_out=train.transitionProbability_zeroOrder(corpus)            
            emissionProbability_out=train.emissionProbability(corpus,configNo)            
            print("Training completed!!")
            dataset="test"            
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            print(name)
            test=decode_0.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,tagCount_out) 
            test.decode() 
        elif int(tagger)==1:
            print("Start of training for first order ")
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset) 
            #print(corpus)
            tagCount_out=train.tagCount(corpus)            
            transitionProbability_out,forwardtagcount_out=train.transitionProbability_firstOrder(corpus) 
            #print(transitionProbability_out)
            emissionProbability_out=train.emissionProbability(corpus,configNo) 
            #print(emissionProbability_out)
            print("Training completed!!") 
            dataset="test"
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            test=decode_1.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,forwardtagcount_out) 
            test_output=test.decode()
        elif int(tagger)==2:
            print("Start of training for second order ")
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset)
            tagCount_out=train.tagCount(corpus)                                 
            transitionProbability_out,forwardtagcount_out=train.transitionProbability_secondOrder(corpus)          
            emissionProbability_out=train.emissionProbability(corpus,configNo)            
            print("Training completed!!")
            dataset="test"
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            test=decode_2.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,forwardtagcount_out)
            test_output=test.decode()                   
        else:
            print("Not a valid input")      
        print("--- %s seconds is total execution time ---" % (time.time() - start_time))
    else:
        print("Invalid corpus name")
    

#####run1#####
#main("conf 0",2,"penn")#--- 14142.936678171158 seconds is total execution time ---
#main("conf 2",2,"penn")#--- 15974.81058883667 seconds is total execution time ---
#main("conf 3",2,"penn")#--- 2608.144533395767 seconds is total execution time ---
#main("conf 4",2,"penn")#--- 7300.396208047867 seconds is total execution time ---
#main("conf 5",2,"penn")#--- 9066.740124940872 seconds is total execution time ---
#main("conf 6",2,"penn")#--- 20.417510271072388 seconds is total execution time ---


#####run2#####
# main("conf 0",0,"genia")#--- 1827.0823800563812 seconds is total execution time ---
# main("conf 1",0,"genia")#--- 217.66432118415833 seconds is total execution time ---
# main("conf 2",0,"genia")#--- 1609.3500781059265 seconds is total execution time ---
# main("conf 3",0,"genia")#--- 215.25380992889404 seconds is total execution time ---
# main("conf 4",0,"genia")#--- 1159.6458036899567 seconds is total execution time ---
# main("conf 5",0,"genia")#--- 1287.1840150356293 seconds is total execution time ---
# main("conf 6",0,"genia")#--- 1.9353759288787842 seconds is total execution time ---

#####run3#####
#main("conf 0",1,"genia")#--- 25307.225355148315 seconds is total execution time ---
#main("conf 1",1,"genia")#--- 2939.4099950790405 seconds is total execution time ---
#main("conf 2",1,"genia")#--- 21530.320410966873 seconds is total execution time ---
#main("conf 3",1,"genia")#--- 2847.127960205078 seconds is total execution time ---
#main("conf 4",1,"genia")#--- 17021.079292058945 seconds is total execution time ---
#main("conf 5",1,"genia")#--- 15868.311786174774 seconds is total execution time ---
#main("conf 6",1,"genia")#--- 9.758394956588745 seconds is total execution time ---  

#####run4#####
# main("conf 0",2,"genia")#--- 269900.1518688202 seconds is total execution time ---
# main("conf 1",2,"genia")#--- 32231.797437667847 seconds is total execution time ---
# main("conf 2",2,"genia")#--- 224892.63014292717 seconds is total execution time ---
# main("conf 3",2,"genia")#--- 31307.679008245468 seconds is total execution time ---
# main("conf 4",2,"genia")#--- 173184.9062230587 seconds is total execution time ---
# main("conf 5",2,"genia")#--- 184795.68603515625 seconds is total execution time ---
# main("conf 6",2,"genia")#--- 146.78194308280945 seconds is total execution time ---  

#####run5#####
# main("conf 0",2,"conll")#--- 79419.68366408348  seconds is total execution time ---
# main("conf 1",2,"conll")#--- 13260.74667596817 seconds is total execution time ---
# main("conf 2",2,"conll")#--- 80154.92458295822 seconds is total execution time ---
# main("conf 3",2,"conll")#--- 13826.536455869675 seconds is total execution time ---
# main("conf 4",2,"conll")#--- 53837.86930656433 seconds is total execution time ---
# main("conf 5",2,"conll")#--- 184795.68603515625 seconds is total execution time ---
# main("conf 6",2,"conll")#--- 76.394278049469 seconds is total execution time --- 

#####run6#####
# main("conf 0",1,"conll")#--- 9570.50050663948  seconds is total execution time ---
# main("conf 1",1,"conll")#--- 1552.5910341739655 seconds is total execution time ---
# main("conf 2",1,"conll")#--- 9664.912146568298 seconds is total execution time ---
# main("conf 3",1,"conll")#--- 1605.5749816894531 seconds is total execution time ---
# main("conf 4",1,"conll")#--- 6272.126628398895 seconds is total execution time ---
# main("conf 5",1,"conll")#--- 7218.007497787476 seconds is total execution time ---
# main("conf 6",1,"conll")#--- 8.419584035873413 seconds is total execution time --- 


#####run7#####
# main("conf 0",0,"conll")#--- 732.5698211193085  seconds is total execution time ---
# main("conf 1",0,"conll")#--- 122.84874296188354 seconds is total execution time ---
# main("conf 2",0,"conll")#--- 738.5211818218231 seconds is total execution time ---
# main("conf 3",0,"conll")#--- 128.49172043800354 seconds is total execution time ---
# main("conf 4",0,"conll")#--- 487.89725399017334 seconds is total execution time ---
# main("conf 5",0,"conll")#--- 566.2461538314819 seconds is total execution time ---
# main("conf 6",0,"conll")#--- 1.6782968044281006 seconds is total execution time ---


#####run8#####
# main("conf 0",1,"penn")#--- 2824.0300137996674 seconds is total execution time ---
# main("conf 1",1,"penn")#--- 486.9670386314392 seconds is total execution time ---
# main("conf 2",1,"penn")#--- 3104.912937641144 seconds is total execution time ---
# main("conf 3",1,"penn")#--- 501.7296185493469 seconds is total execution time ---
# main("conf 4",1,"penn")#--- 1415.215912103653 seconds is total execution time ---
# main("conf 5",1,"penn")#--- 1790.9135053157806 seconds is total execution time ---
# main("conf 6",1,"penn")#--- 3.563797950744629 seconds is total execution time ---

# #####run9#####
# main("conf 0",0,"penn")#--- 206.678955078125 seconds is total execution time ---
# main("conf 1",0,"penn")#--- 36.22237277030945 seconds is total execution time ---
# main("conf 2",0,"penn")#--- 217.5861291885376 seconds is total execution time ---
# main("conf 3",0,"penn")#--- 37.59640407562256 seconds is total execution time ---
# main("conf 4",0,"penn")#--- 107.1867208480835 seconds is total execution time ---
# main("conf 5",0,"penn")#--- 136.36419415473938 seconds is total execution time ---
# main("conf 6",0,"penn")#--- 0.7129175662994385 seconds is total execution time ---


# #####run10#####
# main("conf 0",0,"penn","conll")#--- 275.3447937965393 seconds is total execution time ---
# main("conf 1",0,"penn","conll")#--- 47.96864914894104 seconds is total execution time ---
# main("conf 2",0,"penn","conll")#--- 294.4354028701782 seconds is total execution time ---
# main("conf 3",0,"penn","conll")#--- 48.3225569725036 seconds is total execution time ---
# main("conf 4",0,"penn","conll")#--- 142.81923389434814 seconds is total execution time ---
# main("conf 5",0,"penn","conll")#--- 180.71910309791565 seconds is total execution time ---
# main("conf 6",0,"penn","conll")#--- 0.6090579032897949 seconds is total execution time ---

# #####run11#####
# main("conf 0",0,"conll","penn")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",0,"conll","penn")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",0,"conll","penn")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",0,"conll","penn")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",0,"conll","penn")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",0,"conll","penn")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",0,"conll","penn")#--- 0.5748412609100342 seconds is total execution time ---


# #####run12#####
# main("conf 0",0,"penn","genia")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",0,"penn","genia")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",0,"penn","genia")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",0,"penn","genia")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",0,"penn","genia")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",0,"penn","genia")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",0,"penn","genia")#--- 0.5748412609100342 seconds is total execution time ---

# #####run13#####
# main("conf 0",0,"genia","penn")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",0,"genia","penn")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",0,"genia","penn")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",0,"genia","penn")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",0,"genia","penn")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",0,"genia","penn")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",0,"genia","penn")#--- 0.5748412609100342 seconds is total execution time ---


# #####run14#####
# main("conf 0",0,"conll","genia")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",0,"conll","genia")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",0,"conll","genia")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",0,"conll","genia")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",0,"conll","genia")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",0,"conll","genia")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",0,"conll","genia")#--- 0.5748412609100342 seconds is total execution time ---

# #####run15#####
# main("conf 0",0,"genia","conll")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",0,"genia","conll")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",0,"genia","conll")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",0,"genia","conll")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",0,"genia","conll")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",0,"genia","conll")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",0,"genia","conll")#--- 0.5748412609100342 seconds is total execution time ---


# #####run16#####
# main("conf 0",2,"penn","conll")#--- 275.3447937965393 seconds is total execution time ---
# main("conf 1",2,"penn","conll")#--- 47.96864914894104 seconds is total execution time ---
# main("conf 2",2,"penn","conll")#--- 294.4354028701782 seconds is total execution time ---
# main("conf 3",2,"penn","conll")#--- 48.3225569725036 seconds is total execution time ---
# main("conf 4",2,"penn","conll")#--- 142.81923389434814 seconds is total execution time ---
# main("conf 5",2,"penn","conll")#--- 180.71910309791565 seconds is total execution time ---
# main("conf 6",2,"penn","conll")#--- 0.6090579032897949 seconds is total execution time ---

# #####run17#####
# main("conf 0",2,"conll","penn")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",2,"conll","penn")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",2,"conll","penn")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",2,"conll","penn")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",2,"conll","penn")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",2,"conll","penn")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",2,"conll","penn")#--- 0.5748412609100342 seconds is total execution time ---


# #####run18#####
# main("conf 0",2,"penn","genia")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",2,"penn","genia")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",2,"penn","genia")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",2,"penn","genia")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",2,"penn","genia")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",2,"penn","genia")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",2,"penn","genia")#--- 0.5748412609100342 seconds is total execution time ---

# #####run19#####
# main("conf 0",2,"genia","penn")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",2,"genia","penn")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",2,"genia","penn")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",2,"genia","penn")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",2,"genia","penn")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",2,"genia","penn")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",2,"genia","penn")#--- 0.5748412609100342 seconds is total execution time ---


# #####run20#####
# main("conf 0",2,"conll","genia")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",2,"conll","genia")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",2,"conll","genia")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",2,"conll","genia")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",2,"conll","genia")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",2,"conll","genia")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",2,"conll","genia")#--- 0.5748412609100342 seconds is total execution time ---

# #####run21#####
# main("conf 0",2,"genia","conll")#--- 218.13787293434143 seconds is total execution time ---
# main("conf 1",2,"genia","conll")#--- 36.430442810058594 seconds is total execution time ---
# main("conf 2",2,"genia","conll")#--- 210.3587679862976 seconds is total execution time ---
# main("conf 3",2,"genia","conll")#--- 37.81561875343323 seconds is total execution time ---
# main("conf 4",2,"genia","conll")#--- 137.3475079536438 seconds is total execution time ---
# main("conf 5",2,"genia","conll")#--- 158.36580610275269 seconds is total execution time ---
# main("conf 6",2,"genia","conll")#--- 0.5748412609100342 seconds is total execution time ---


# #####run22#####
# main("conf 0",1,"penn","conll")#--- 3902.913197994232 seconds is total execution time ---
# main("conf 1",1,"penn","conll")#--- 695.2444968223572 seconds is total execution time ---
# main("conf 2",1,"penn","conll")#--- 4415.135587930679 seconds is total execution time ---
# main("conf 3",1,"penn","conll")#--- 737.0762710571289 seconds is total execution time ---
# main("conf 4",1,"penn","conll")#--- 2028.976662158966 seconds is total execution time ---
# main("conf 5",1,"penn","conll")#--- 2593.6222100257874 seconds is total execution time ---
# main("conf 6",1,"penn","conll")#--- 4.356599807739258 seconds is total execution time ---

# #####run23#####
# main("conf 0",1,"conll","penn")#--- 3166.9177689552307 seconds is total execution time ---
# main("conf 1",1,"conll","penn")#--- 520.5246539115906 seconds is total execution time ---
# main("conf 2",1,"conll","penn")#--- 3146.1290559768677 seconds is total execution time ---
# main("conf 3",1,"conll","penn")#--- 520.7189769744873 seconds is total execution time ---
# main("conf 4",1,"conll","penn")#--- 2023.201908826828 seconds is total execution time ---
# main("conf 5",1,"conll","penn")#--- 2340.918256044388 seconds is total execution time ---
# main("conf 6",1,"conll","penn")#--- 2.3893120288848877 seconds is total execution time ---


#####run24#####
# main("conf 0",1,"penn","genia")#--- 11732.791595220566 seconds is total execution time ---
# main("conf 1",1,"penn","genia")#--- 1823.5606000423431 seconds is total execution time ---
# main("conf 2",1,"penn","genia")#--- 11390.103175878525 seconds is total execution time ---
# main("conf 3",1,"penn","genia")#--- 1927.9312121868134 seconds is total execution time ---
# main("conf 4",1,"penn","genia")#--- 4756.91988492012 seconds is total execution time ---
# main("conf 5",1,"penn","genia")#--- 6290.496965885162 seconds is total execution time ---
# main("conf 6",1,"penn","genia")#--- 9.632323980331421 seconds is total execution time ---

# #####run25#####
# main("conf 0",1,"genia","penn")#--- 5643.633567094803 seconds is total execution time ---
# main("conf 1",1,"genia","penn")#--- 691.5237262248993 seconds is total execution time ---
# main("conf 2",1,"genia","penn")#--- 4868.635155916214 seconds is total execution time ---
# main("conf 3",1,"genia","penn")#--- 677.4127871990204 seconds is total execution time ---
# main("conf 4",1,"genia","penn")#--- 2775.533472776413 seconds is total execution time ---
# main("conf 5",1,"genia","penn")#--- 2888.3404738903046 seconds is total execution time ---
# main("conf 6",1,"genia","penn")#--- 2.8230488300323486 seconds is total execution time ---


# #####run26#####
# main("conf 0",1,"conll","genia")#--- 18110.913549900055 seconds is total execution time ---
# main("conf 1",1,"conll","genia")#--- 3304.8499472141266 seconds is total execution time ---
# main("conf 2",1,"conll","genia")#-- 18431.422095775604 seconds is total execution time ---
# main("conf 3",1,"conll","genia")#--- 2960.5194828510284 seconds is total execution time ---
# main("conf 4",1,"conll","genia")#--- 8858.058780908585 seconds is total execution time ---
# main("conf 5",1,"conll","genia")#--- 11186.920429944992 seconds is total execution time ---
# main("conf 6",1,"conll","genia")#--- 9.659415006637573 seconds is total execution time ---

# #####run27#####
# main("conf 0",1,"genia","conll")#--- 12445.687447786331 seconds is total execution time ---
# main("conf 1",1,"genia","conll")#--- 1533.5909957885742 seconds is total execution time ---
# main("conf 2",1,"genia","conll")#--- 11101.632594823837 seconds is total execution time ---
# main("conf 3",1,"genia","conll")#--- 1505.618490934372 seconds is total execution time ---
# main("conf 4",1,"genia","conll")#--- 6057.243942260742 seconds is total execution time ---
# main("conf 5",1,"genia","conll")#--- 6542.662256002426 seconds is total execution time ---
# main("conf 6",1,"genia","conll")#--- 5.05007791519165 seconds is total execution time ---