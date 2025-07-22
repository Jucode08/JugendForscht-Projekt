from deep_translator import GoogleTranslator

# Use any translator you like, in this example GoogleTranslator
# translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome") 
translated = GoogleTranslator(source='auto', target='en').translate("Mach weiter so, du bist großartig") 
print(translated)