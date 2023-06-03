# JavaScript是什么
JavaScript（缩写：JS）是一门完备的动态编程语言。当应用于 HTML 文档时，可为网站提供动态交互特性.

开发者们基于 JavaScript 核心编写了大量实用工具:
- 浏览器应用程序接口（API）—— 浏览器内置的 API 提供了丰富的功能，比如：动态创建 HTML 和设置 CSS 样式、从用户的摄像头采集处理视频流、生成 3D 图像与音频样本等等。
- 第三方 API —— 让开发者可以在自己的站点中整合其他内容提供者（Twitter、Facebook 等）提供的功能。
- 第三方框架和库 —— 用来快速构建网站和应用。
# hello world
body中插入
```
<script src="scripts/main.js" defer></script>
```
main.js中写入
```
let myHeading = document.querySelector('h1');
myHeading.textContent = 'Hello world!';
```
用 querySelector() 函数获取标题的引用，并把它储存在 myHeading 变量中.把 myHeading 变量的属性textContent （标题内容）修改为“Hello world!” 

#将 JavaScript 代码放在 HTML 页面的底部附近通常是最好的策略
# JavaScript 快速入门
## 变量
1. 定义
- 关键字 let 或 var
```
let myVariable;
```
2. 赋值
```
#赋值
myVariable = '李雷';
#在定义时直接赋值
let myVariable = '李雷';
#通过变量名获得其值
myVariable;
```
数据类型
|变量|解释|示例|
|----|----|----|
|String|字符串，单双括起|let myVariable = '李雷';|
|Number|数字|let myVariable = 10;|
|Boolean|布尔值true/false|let myVariable = true;|
|Array|数组|let myVariable = [1, '李雷', '韩梅梅', 10];元素引用方法：myVariable[0], myVariable[1]|
|Object|对象：JavaScript 里一切皆对象，一切皆可储存在变量里|let myVariable = document.querySelector('h1');以及上面所有示例都是对象。|
## 注释
```
/*
这里的所有内容
都是注释。
*/
// 这是一行注释
```
## 运算符
|运算符|解释|符号|示例|
|---|---|---|---|
|加|将两个数字相加，或拼接两个字符串。|+|6 + 9;"Hello " + "world!";|
|减、乘、除||-, *, /|9 - 3;8 * 2; 9 / 3;|
|赋值运算符||=||
|等于|返回一个 true/false （布尔）值|===|myVariable === 4; // false|
|不等于||!==||
|取非||!||
## 条件语句
```
if(){

}
else{

}
```
## 函数(Fuction)
- document.querySelector 和alert 是浏览器内置的函数.alert() 函数在浏览器窗口内弹出一个警告框
```
function name(parm1,param2){
    return ;
}
```
## 事件
```
document.querySelector("html").addEventListener("click", function () {
  alert("别戳我，我怕疼。");
});
```
此处传入了匿名函数function ()
```
document.querySelector('html').addEventListener('click', () => {
  alert('别戳我，我怕疼。');
});
```
箭头函数使用 () => 代替 function ()
# 发布网站
## 获取主机服务和域名
- 主机服务:在主机服务提供商的 Web 服务器上租用文件空间。将你网站的文件上传到这里，然后服务器会提供 Web 用户需求的内容。
- 域名——一个可以让人们访问的独一无二的地址，比如 ```http://www.mozilla.org```,你可以从域名注册商租借域名。
-  文件传输协议程序:将网站文件上传到服务器
## 使用在线工具如 GitHub 或 Google App Engine
# 通过 GitHub 发布