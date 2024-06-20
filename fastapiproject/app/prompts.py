
labels = ['Extremely Positive', 'Positive', 'Neutral', 'Negative', 'Extremely Negative']

prompt_instruction = f"""1.You are a Classifier. Your task is to assign one label out of the given labels: {labels} to the given text.
                        2. Please ONLY PROVIDE LABEL as the response.
                        3. Examples of the text and labels are given below:
                        Text: 'We're here to provide a safe shopping experience for our customers and a healthy environment for our associates and community!Online orders can be placed here' 
                        Label: ['Extremely Positive'] 
                        Text: 'my wife works retail;a customer came in yesterday, coughing everywhere, saying they have CoVid-19. They requested a deep clean of the store - her company objected to due to cost, recommending the team spray disinfectant&amp;clean themselves. we're gonna die/get sick due to capitalism'
                        Label: ['Negative']
                        Text: 'Curious,  do we think retail shoppers will do a lot of online shopping bc they're home and unable to go out or do we think everyone is too spooked to get that extra pair of shoes?'
                        Label: ['Positive']
                        Text: 'Both the masks made for medical personnel and for consumer purchase require a once-obscure material called melt-blown fabric'
                        Label: ['Neutral']
                        Text : 'Bought a house during Covid-19 panic. Didn’t think to buy food for the house. Tragic'
                        Label: ['Extremely Negative']
                        4. You can use the above examples to assign a label to the given text."""
