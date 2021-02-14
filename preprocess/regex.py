import re

text_to_search = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789

Ha HaHa

Metacharacters (Need to be escaped):
.^$*+?{}[]\|()

clickthis.com

321-555-4321
123.555.1234

Mr. Jones
Mr Mistake
Ms Misunderstanding
Mrs. Milf
Mr. Q

cat
mat
pat
bat

KakkaPylly@gmail.com
kakka.pylly@university.edu
kakka-77-pylly@my-work.net

https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov

'''

#simple example
pattern = re.compile(r'clickthis\.com')

matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)
    idx = match.span()
    print(idx)

print(text_to_search[idx[0]:idx[1]])

#more advanced pattern

# Meta characters:
# . --> matches all characters except whitespace
# \d --> any digit between 0 and 9
# \w --> word character (a-z, A-Z, 0-9, _)
# \s --> Whitespace (space, tab, newline)

# \b --> word boundary
# \uppercase --> negation
# ^ --> beginning of a string
# $ --> end of a string

# [] --> matches characters in brackets
# [^ ] --> matches characters not in brackets

# * --> 0 or more
# + --> 1 or more
# ? --> 0 or 1
# {x} --> x amount
# {x,y} --> range (min, max)


# phone numbers
#pattern = re.compile(r'\d{3}[-.]\d{3}[-.]\d{4}')

# not bat
#pattern = re.compile(r'[^b]at')

# names
#pattern = re.compile(r'M(r|s|rs)\.?\s[A-Z]\w*')

# emails
#pattern = re.compile(r'[\w.-]+@[\w-]+\.(com|edu|net)')

# urls
#pattern = re.compile(r'http[s]*://[www.]*[\w]+\.(com|gov|net)')
pattern = re.compile(r'https?://(www\.)?(\w+)(\.(com|gov|net))')

# flags
# pattern = re.compile(r'abc', re.IGNORECASE) # ignores case

matches = pattern.finditer(text_to_search)
# matches = pattern.findall(text_to_search) # returns matches as a list of strings
# matches = pattern.match(text_to_search) # returns first match if any (in the beginning of a string)
# matches = pattern.search(text_to_search) # returns first if any (whole string)

# for finditer
for match in matches:
    print(match)
    # print(match.group(2))
    idx = match.span()
    print(text_to_search[idx[0]:idx[1]])

# for findall
'''
for mathc in matches:
    print(match)
'''

# for match and search
# print(matches)