# Implementing a spam filter using Naive Bayes and TextAnalysis.jl

Hello! Today I'm gonna tell you more about what I did to make a spam filter using Naive Bayes to detect data from a csv file, along with the use of [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) on [Julia](https://julialang.org/).

I started off by looking at the docs off TextAnalysis.jl to understand more about how exactly the [NaiveBayes Classifier](https://juliatext.github.io/TextAnalysis.jl/latest/classify/) works.

```julia
using TextAnalysis: NaiveBayesClassifier, fit!, predict
m = NaiveBayesClassifier([:legal, :financial])
fit!(m, "this is financial doc", :financial)
fit!(m, "this is legal doc", :legal)
predict(m, "this should be predicted as a legal document")
```

I ran the example from the docs and I learned that the function NaiveBayesClassifier takes in the argument of an array of possible classes that the concerned data could perhaps belong to.

In this case, it was `:legal` and `:financial`. I also learned that we have train the model by fitting the concerned data with the `fit!` function where it takes in the arguments of the model itself, the string of data we are trying to train, and the class that data belongs to. The data here is a string, for example `“this is financial doc”` and the class it belongs to is, in this case, `:financial`.

Finally, I learned that the `predict` function allows us to enter a string of data and uses the NaiveBayesClassifier algorithm to predict what class the string belongs to, based on the strings of data trained before using the `fit!` function. The `predict` function takes in the arguments of the model itself as well as the string of data we are trying to predict.

My first approach to the problem was that I thought it would be a good idea to first import all of the data. As I have experience with the packages [CSV.jl](https://github.com/JuliaData/CSV.jl) and [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl), I am familiar with the importing of the data. 

```julia
using CSV, DataFrames
spamdata = DataFrame(CSV.read("spam.csv"; allowmissing=:none))
```

```julia
julia> showall(spamdata)
5572×5 DataFrame
│ Row  │ v1     │
│      │ String │
├──────┼────────┤
│ 1    │ ham    │
│ 2    │ ham    │
│ 3    │ spam   │
│ 4    │ ham    │
│ 5    │ ham    │
│ 6    │ spam   │
│ 7    │ ham    │
│ 8    │ ham    │
│ 9    │ spam   │
│ 10   │ spam   │
```

This image below shows the structure of the original `.csv` file with the data.

<img width="364" alt="img" src="https://user-images.githubusercontent.com/28962234/72909063-57db4800-3d2e-11ea-8a4b-c2e49f96f341.png">

The csv file has two columns. `v2` is the string of the data we want to use to train and `v1` is the class of the particular string of data, corresponding to `v2`.

I want a way to loop through every single row of the file and split the ham data in one condition and spam data in another, such that when it comes to training, I will be able to train the `ham` data using the `fit!` function, which requires me to specify the class of that data.

```julia
for row in eachrow(spamdata)
    if row.v1 == "ham"
        println("ham")
    elseif row.v1 == "spam"
        println("spam")
    end
end
```
This was a success and I ended up having hams and spams being printed!

```julia
ham
spam
ham
spam
ham
ham
spam
spam
ham
spam
⋮
```

Now that it is out of the way, I can define my model using the NaiveBayesClassifier function. I want to define 2 classes, `:ham` and `:spam`.

```julia
using TextAnalysis: NaiveBayesClassifier, fit!, predict
m = NaiveBayesClassifier([:ham, :spam])
```
Next, I want to start training my model. As we saw from the original `.csv` file’s structure, `v2` is the string we are trying to train. Combining the `for` loop with the `fit!` function, I did the following to try and train all the data we have available.

```julia
using CSV, DataFrames
using TextAnalysis: NaiveBayesClassifier, fit!, predict

spamdata = DataFrame(CSV.read("spam.csv"; allowmissing=:none))
global m = NaiveBayesClassifier([:ham, :spam])
for row in eachrow(spamdata)
    if row.v1 == "ham"
        fit!(m, row.v2, :ham)
    elseif row.v1 == "spam"
        fit!(m, row.v2, :spam)
    end
end
```
But when I ran that, I got the following error: `LoadError: Base.InvalidCharError{Char}('\xe5\xa3')`

It was then when I realized that the dataset had invalid characters in certain strings in the `v2` column. To eliminate this error, we need to filter out the unsupported characters using the following function: `filter(isvalid, <string>)`

```julia
for row in eachrow(spamdata)
    if row.v1 == "ham"
        fit!(m, filter(isvalid, row.v2), :ham)
    elseif row.v1 == "spam"
        fit!(m, filter(isvalid, row.v2), :spam)
    end
end
```

When I replaced the string `row.v2` with `filter(isvalid, row.v2)` and ran the program once again, no errors came up. Therefore the model was successfully trained and so far so good!

Nothing was actually printed out in the REPL because there was no error and nothing in the program had anything printed out. To fully test that the model is working, we can try and create a prediction and print out the results of the prediction to see whether the training was really done or not.

```julia
prediction1 = predict(m, "hello my name is kfung")
prediction2 = predict(m, "text 31845 to get a free phone")
println(prediction1)
println(prediction2)
```

Here I created two predictions, where the first one looks like a ham message (as it doesn’t look suspicious), and a second one that looks like a spam message to test out the model.

The results I got are below:

```julia
Dict(:spam => 0.013170434049325023, :ham => 0.986829565950675)
Dict(:spam => 0.9892304346396908, :ham => 0.010769565360309069)
```

As we can tell, the predictions were pretty accurate as the first prediction had a `:ham` value close to 1 meaning it is most probably a ham message, and the second prediction had a :`spam value` close to 1 meaning it is most likely a spam message. Exactly what we expected.

But for users who are not familiar with dictionaries or Julia syntax, they may be confused as to what the dictionaries above mean. I modified the code so that it checks for the `:spam` and the `:ham` values in the dictionary and prints out the class of the bigger value of those two.

```julia
prediction = predict(m, "hello my name is kfung")
if prediction[:spam] > prediction[:ham]
    println("spam")
else
    println("ham")
end
```

As we expected, the results of this program was the string `“ham”` because the `:ham` value it predicted was greater than the `:spam` value, therefore it is more likely to be a ham message.

In the end, I wrapped everything in a function that takes a string as an argument so that when  you call the function with a string, it will print out either `“spam”` if the model predicts it as a spam message, or `“ham”` if the model predicts that it is not a spam message.

```julia
  
using CSV, DataFrames
using TextAnalysis: NaiveBayesClassifier, fit!, predict

function checkspam(msg::String)
    spamdata = DataFrame(CSV.read("spam.csv"; allowmissing=:none))
    m = NaiveBayesClassifier([:ham, :spam])
    for row in eachrow(spamdata)
        if row.v1 == "ham"
            fit!(m, filter(isvalid, row.v2), :ham)
        elseif row.v1 == "spam"
            fit!(m, filter(isvalid, row.v2), :spam)
        end
    end
    prediction = predict(m, msg)
    if prediction[:spam] > prediction[:ham]
        println("spam")
    else
        println("ham (not spam)")
    end
end
```

In the end I realized how I was training the model every time a classify a message. The makes the program not very efficient at run time because every time we are trying to predict using the model, it has gone through through the 5600 lines of data all over again. Instead, we can bring the modal outside of the function (so that it is run at the start only once) and store the model in a global variable so that afterwards it can just used the pre-trained model stored in the global variable to classify any other messages.

```julia
using CSV, DataFrames
using TextAnalysis: NaiveBayesClassifier, fit!, predict

spamdata = DataFrame(CSV.read("spam.csv"; allowmissing=:none))
global m = NaiveBayesClassifier([:ham, :spam])
for row in eachrow(spamdata)
    if row.v1 == "ham"
        fit!(m, filter(isvalid, row.v2), :ham)
    elseif row.v1 == "spam"
        fit!(m, filter(isvalid, row.v2), :spam)
    end
end

function checkspam(msg::String)
    prediction = predict(m, msg)
    if prediction[:spam] > prediction[:ham]
        println("spam")
    else
        println("ham (not spam)")
    end
end
```

### Thanks so much for reading! This blog was written by kfung.
