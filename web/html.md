# 基本格式
```
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>mytitle</title>
  </head>
  <body>
    待填充的内容
  </body>
</html>
```
- ```<html></html>``` — ```<html>``` 元素。该元素包含整个页面的内容，也称作根元素。
- ```<head></head>``` — ```<head>``` 元素。该元素的内容对用户不可见，其中包含例如面向搜索引擎的搜索关键字（keywords）、页面描述、CSS 样式表和字符编码声明等。
- ```<meta charset="utf-8">``` — 该元素指定文档使用 UTF-8 字符编码，UTF-8 包括绝大多数人类已知语言的字符。基本上 UTF-8 可以处理任何文本内容，还可以避免以后出现某些问题，没有理由再选用其他编码。
- ```<title></title>``` — ```<title>``` 元素。该元素设置页面的标题，显示在浏览器标签页上，也作为收藏网页的描述文字。
-``` <body></body>``` — ```<body>``` 元素。该元素包含期望让用户在访问页面时看到的内容，包括文本、图像、视频、游戏、可播放的音轨或其他内容。
# html元素组成
```
<p class="editor-note">my cat<\p>
```
1. 开始标签（opening tag）：包含元素的名称，被<,>所包围，例子中```<p>```。
2. 结束标签（closing tag）:与开始标签相似，只是其在元素名之前包含了一个斜杠\，```<\p>```
3. 内容（content）：例子中```my cat```
4. 元素（Element）:开始标签、结束标签与内容相结合，便是一个完整的元素
5. 属性（Attribute）:例子中```class="editor-note"```
# 图像
```<img src="" alt="" />```
- src:地址
- alt:图像的描述内容，用于当图像不能被用户看见时显示
# 标记文本
## 标题（Heading）
- 包括六个级别的标题,h1-h6
```
<h1>主标题</h1>
<h2>顶层标题</h2>
```
## 段落（Paragraph）
```<p> ```元素
```
<p>这是一个段落</p>
```
## 列表(List)
### 1. 无序列表(Unordered List)```<ul>``` 元素包围
### 2. 有序列表(Ordered List)```<ol>``` 元素包围
列表的每个项目用一个列表项目（List Item）元素 ```<li> ```包围。
```
<ul>
  <li>technologists</li>
  <li>thinkers</li>
  <li>builders</li>
</ul>
```
# 链接```<a>```元素(anchor)
```
<a href="https://www.baidu.com">baidu</a>
```
- 如果网址开始部分省略了 https:// 或者 http://，可能会得到错误的结果