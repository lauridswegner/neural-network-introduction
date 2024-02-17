# neural-network-introduction

**Disclaimer:**
The implementation of the neuralNetwork is from [Tariq Rashid](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork).

***

<img width="649" alt="image" src="https://github.com/lauridswegner/neural-network-introduction/assets/128291909/36fe19bf-586f-4f13-b9a5-3c41e15ec26c">

***

## Explanation

This project is my little crappy introduction into neuralNetworks and the use of one to classify humanwritten digits.

## To-Do's

- [x] make neuralNetwork weights storable for use after restart
- [x] create a canvas for drawing digits
    - [x] make them detectable by the neuralNetworks query, forcing a prediction to be made
- [ ] increase accuracy
    - [ ] input data is still not simillar enough to mnist data
    - [ ] include rotated digits on training-data
- [ ] make a second script focussing on the training and finetuning of the network

## How to run

- run `pip install .` in the root directory of the project, installing the neuralNetwork package
- afterwards run `app.py` to enter the gui and let it start guessing numbers
