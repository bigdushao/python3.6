# coding:utf-8
"""
使用python的正则表达式模块进行一些简单的正则练习
"""
import re

# 将正则表达式编译成Pattern对象
pattern = re.compile(r'hello')

# 使用pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('hello world')

if match:
    print(match.group(), match.groupdict(), match.groups())
'''
re.compile(strPattern[, flag])
    这个方法是Pattern类的工厂方法，用于将字符串形式的正则表达式编译为Pattern对象，第二个参数flag是匹配模式，取值可以使用按位或运算
    符'|'表示同时生效，比如re.I | re.M。
    re.X 通过给予更灵活的格式以便将正则表达式写的更加灵活便于理解， 在正则表达式中添加注释
    另外，你也可以在regex字符串中指定模式，比如re.compile('pattern', re.I | re.M)与re.compile('(?im)pattern')是等价的。 
    
    (?P<name>...)        分组的命名模式，取此分组中的内容时可以使用索引也可以使用name
    (?P=name)            分组的引用模式，可在同一个正则表达式用引用前面命名过的正则
    (?#...)              注释，不影响正则表达式其它部分
    (?(id/name)yes|no)   若前面指定id或name的分区匹配成功则执行yes处的正则，否则执行no处的正则
Match对象是一侧匹配的结果，包含了很多关于此次匹配的信息，可以使用Match提供的可读属性或方法来获取这些信息
    属性:string:匹配时使用的文本
        re:匹配时使用的Pattern对象
        pos:文本中正则表达式开始搜索的索引。
        endpos:文本中正则表达式结束搜索的索引。
        lastindex:最后一个被捕获的分组在文本中的索引。如果没有被捕获的分组。将为None
        lastgroup:最后一个被捕获的分组的别名，如果这个分组没有别名或者没有被捕获的分组。将为None
    方法:group([group1,....]),获得一个或多个分组截获的字符串，指定多个参数时将以元组新办公室返回，group1可以使用编号也可以使用别名；编号0代表整个匹配的子串；
        不填写参数时，返回group(0)；没有截获字符串的组返回None；截获了多次的组返回最后一次截获的子串。
        groups([default])：以元组形式返回全部分组截获的字符串。相当于调用group(1,2,…last)。default表示没有截获字符串的组以这个值替代，默认为None。
        groupdict([default])：返回以有别名的组的别名为键、以该组截获的子串为值的字典，没有别名的组不包含在内。default含义同上
        start([group])：返回指定的组截获的子串在string中的起始索引（子串第一个字符的索引）。group默认值为0。
        end([group]):返回指定的组截获的子串在string中的结束索引（子串最后一个字符的索引+1）。group默认值为0。
        span([group]):返回(start(group), end(group))。
        expand(template):将匹配到的分组代入template中然后返回。template中可以使用\id或\g<id>、\g<name>引用分组，但不能使用编号0。
            \id与\g<id>是等价的；但\10将被认为是第10个分组，如果你想表达\1之后是字符'0'，只能使用\g<1>0。
            
        Pattern：
        Pattern对象是一个编译好的正则表达式。通过pattern提供的一系列方法可以对文本进行匹配查找
        Pattern不能直接实例化，必须通过re.compile()进行构造
        Pattern提供了几个可读属性，用于获取表达式的相关信息:
            pattern：编译时用的表达式字符串
            flags:编译时用的匹配模式，数字形式
            groups:表达式中的分组的数量。
            groupindex:表达式中有别名的组的别名为键，以改组对应的编号为值的字典。没有别名的组不包含在内
        实例方法:
        match(string[, pos[, endpos]]) | re.match(pattern, string[, flags]);这个方法将从string的pos下标处起尝试匹配pattern
        如果pattern结束时仍可匹配，则返回一个Match对象；如果匹配过程中pattern无法匹配，或者匹配未结束就已到达endpos，则返回None。 
        pos和endpos的默认值分别为0和len(string)；re.match()无法指定这两个参数，参数flags用于编译pattern时指定匹配模式。 
        注意：这个方法并不是完全匹配。当pattern结束时若string还有剩余字符，仍然视为成功。想要完全匹配，可以在表达式末尾加上边界匹配符'$'
        
        search(string[, pos[, endpos]]) | re.search(pattern, string[, flags]):这个方法用于查找字符串中可以匹配成功的子串。
        从string的pos下标处起尝试匹配pattern，如果pattern结束时仍可匹配，则返回一个Match对象；若无法匹配，则将pos加1后重新尝试匹配；
        直到pos=endpos时仍无法匹配则返回None。 
        pos和endpos的默认值分别为0和len(string))；re.search()无法指定这两个参数，参数flags用于编译pattern时指定匹配模式。
        
        split(string[, maxsplit]) | re.split(pattern, string[, maxsplit]):按照能够匹配的子串将string分割后返回列表。maxsplit用于制定最大分割次数，不指定将全部分割
        
        findall(string[, pos[, endpos]]) | re.findall(pattern, string[, endpos]):搜索string，以列表形式返回全部能匹配的子串
        
        finditer(string[, pos[, endpos]]) | re.finditer(pattern, string[, flags]):搜索string，返回一个顺序访问每一个匹配结果（Match对象）的迭代器。 
        
        sub(repl, string[, count]) | re.sub(pattern, reple, string[, count]):使用repe替换string中每一个匹配的子串后返回替换后的字符串
        当repl是一个字符串时，可以使用\id或\g<id>、\g<name>引用分组，但不能使用编号0。 
        当repl是一个方法时，这个方法应当只接受一个参数（Match对象），并返回一个字符串用于替换（返回的字符串中不能再引用分组）。 
        count用于指定最多替换次数，不指定时全部替换。 
        
        subn(repl, string[, count]) |re.sub(pattern, repl, string[, count]): 
        返回 (sub(repl, string[, count]), 替换次数)。
'''
print(re.match(r'a.bc|abc', 'adbc')) # 表示或
print(re.match(r'a[bcdef]{1,2}a', 'abca')) # 字符集中的匹配之能够匹配一次，使用{1, 2} 字符集中的字符匹配一次或者多次
print(re.match(r'(?P<id>abc){2}', 'abcabc'))

a = re.compile(r"""\d + # the integral part
               \. # the decimal point
               \d* # some fractional digits""", re.X)
b = re.compile(r"\d+\.\d*")

# m = re.match(r'''(\w+) # 分组匹配，匹配所有的字符串 + 空格
#             (\w+) # 匹配所有的字符串
#             (?P<a>.*) # 分组匹配字符差un之后的其他的内容，将匹配的结果起了一个别名
#             ''', re.X, 'hello world!')

m = re.match(r'(\w+) (\w+)(?P<a>.*)', 'hello world!')
print("m.string", m.string)
print("m.re", m.re)
print('m.group', m.group())
print('m.group(1)', m.group(1))
print('m.group(2)', m.group(2))
print('m.group(3)', m.group(3))
print('m.groups', m.groups())
print('m.groupdict', m.groupdict())
print("m.endpos", m.endpos)
print('m.lastindex', m.lastindex)
print('m.lastgroup', m.lastgroup)
print('m.group(1, 2)', m.group(1, 2))
print('m.start(2)', m.start(2))
print('m.end(2)', m.end(2))
print('m.span(2)', m.span(2))

p = re.compile(r'\d+')
print(p.split('one1two2three3four4'))
print(p.findall('one1two2three3four4'))

pp = re.compile(r'(\w+) (\w+)')
s = 'i say , hello world'
print(pp.sub(r'\2 \1', s))
def func(m):
    return m.group(1).title() + ' ' + m.group(2).title()
print(pp.sub(func, s))
