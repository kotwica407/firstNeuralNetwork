words = ["make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail", "receive", "will", "people",
 "report", "addresses", "free", "business", "email", "you", "credit", "your", "font", "000", "money", "hp", "hpl",
 "george", "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
 "cs", "meeting", "original", "project", "re", "edu", "table", "conference"]

chars = [';', '(', '[', '!', '$', '#']


def valuesOfAttribute(list, n):
    x = []
    for i in range(len(list)):
        x.append(list[i][n])
    return x

def longestCapitalSequence(text):
    count = 0
    sequence = []
    for c in text:
        if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            sequence.append(c)
        else:
            if c == " ":
                continue
            else:
                if len(sequence)>count:
                    count = len(sequence)
                sequence = []
    return count

def capitalSequences(text, lengths, s):
    sequence = []
    for c in text:
        if sequence == []:
            if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                sequence.append(c)
            else:
                continue
        else:
            if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                sequence.append(c)
            else:
                if c == " ":
                    continue
                else:
                    lengths.append(len(sequence))
                    s.append(sequence)
                    sequence = []
    s.append(sequence)
    lengths.append(len(sequence))

def numberOfWordInText(text, word):
    temp = tuple(text.split(" "))
    count = 0
    for w in temp:
        if w == word:
            count += 1
    return count

def numberOfCharInText(text, character):
    temp = tuple(text.split(" "))
    count = 0
    for w in temp:
        for c in w:
            if c == character:
                count += 1
    return count

def numberOfCapitalsInText(text):
    temp = tuple(text.split(" "))
    count = 0
    for w in temp:
        for c in w:
            if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                count += 1
    return count


def readdata(inp, out):
    with open('D:/learningPython/trainsets/SpamBaseData/spambase.txt', 'r') as content:
        dane = []
        for linia in content:
            linia = linia.replace("\n", "")
            linia = linia.replace("\r", "")
            temp = tuple(linia.split(" "))
            danet = []
            for i in range(len(temp)):
                danet.append(float(temp[i]))
            dane.append(danet)
        for example in dane:
            inpt = []
            for i in range(len(example)):
                if i == (len(example)-1):
                    out.append([example[i]])
                else:
                    inpt.append(example[i])
            inp.append(inpt)


def calcAttributesOfEmail(dir):
    attributes = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    with open(dir, 'r') as email:
        totalNumberOfWords = 0
        totalNumberOfChars = 0
        sequencesOfCapitals = []
        lenghtsOfCapitalSequences = []
        ltemp = []
        for l in email:
            ltemp = []
            l = l.replace("\n", "")
            l = l.replace("\r", "")
            ltemp = tuple(l.split(" "))
            totalNumberOfWords += len(ltemp)
            for c in ltemp:
                totalNumberOfChars += len(c)
        print(totalNumberOfChars)
        print(totalNumberOfWords)
    with open(dir, 'r') as email:
        for linia in email:
            linia = linia.replace("\n", "")
            linia = linia.replace("\r", "")
            for i in range(len(words)):
                attributes[0][i] += numberOfWordInText(linia, words[i])*100/totalNumberOfWords
            for i in range(len(chars)):
                attributes[0][48 + i] += numberOfCharInText(linia, chars[i])*100/totalNumberOfChars
            attributes[0][56] += numberOfCapitalsInText(linia)
            capitalSequences(linia, lenghtsOfCapitalSequences, sequencesOfCapitals)
        attributes[0][55] = max(lenghtsOfCapitalSequences)
        attributes[0][54] = sum(lenghtsOfCapitalSequences)/len(lenghtsOfCapitalSequences)
    return attributes


#attributes = calcAttributesOfEmail('D:/learningPython/testsets/Mails/Spam/Spam1.txt')

print(len(words))

