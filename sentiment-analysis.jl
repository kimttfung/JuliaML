using TextAnalysis

function setup()
    #opening the txt file containing all the information, and automatically closes it when done
    open("amazonreviews/test.ft.txt") do file
        
        #gathering all the data from the txt file
        global mydata = readlines(file)        
    end
    
    #defining the pre-trained model to be used
    global model = SentimentAnalyzer()
end

function analysis()
    #defining two counters which will count the number of reviews analysed and number of correct analysis respectively
    iterations, correct = 0, 0
    
    #for loop for each line in the txt file
    for num in 1:length(mydata)
        
        #try/catch loop to catch out errors, explained later on
        try
            
            #getting the sentiment value from the txt file, 2 = positive, 1 = negative
            sentiment = parse(Int64,string(mydata[num][10])) 
            
            #getting the review as a string
            sd = StringDocument(mydata[num][12:end])
            
            #removing corrupted characters
            remove_corrupt_utf8!(sd)
            
            #plugging the review into the model to get a floating point number ranging from 0-1, 1 = positive, 0 = negative
            value = model(sd)
            
            #adding one to the number of reviews analysed, to be displayed at the end
            iterations = iterations + 1 
            
            #when both are positive (ie. analysed correctly), add one to the number of correct interpretations
            if value > 0.5 && sentiment == 2
                correct = correct + 1
                
            #when both are negative (ie. analysed correctly), add one to the number of correct interpretations
            elseif value < 0.5 && sentiment == 1
                correct = correct + 1
            end
            
        #catches out the BoundsError problem
        catch
            
            #skips the rest of the iteration to ignore the review and move onto the next (by the for loop)
            continue
        end
    end
    
    #print out number of interpretations
    println("number of interpretations: ", iterations)
    
    #print out number of correct interpretations
    println("number of correct interpretations: ", correct)
    
    #print out accuracy as a percentage (ie. how many of those interpretations were correct)
    println("accuracy: ", (correct/iterations*100),"%")
end

setup()
analysis()

# Results:
# Number of interpretations: 399703
# Number of correct interpretations: 207103
# Accuracy: 51.814222059879455%
