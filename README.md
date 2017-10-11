# StrokeWidthTransform
swt: 基于笔画宽度检测文字


参考： https://github.com/mypetyak/StrokeWidthTransform

## 详见论文
["Detecting Text in Natural Scenes with Stroke Width Transform"](http://cmp.felk.cvut.cz/~cernyad2/TextCaptchaPdf/Detecting%20Text%20in%20Natural%20Scenes%20with%20Stroke%20Width%20Transform.pdf)

## 缺点：耗内存，速度慢，检测结果不好, 代码待改进

## 步骤：

1. 使用canny 边缘检测算子提取图像的边缘
2. 计算图像的x向和y向导数，即图像的梯度。每个像素的梯度表明了最大对比度的方向。对于边缘像素，像素的梯度和边缘的法向量是等价的
3. 对于每个边缘像素，沿着梯度θ的方向移动，直到遇到下一个边缘像素。如果对应的边缘像素的梯度指向了相反的方向（θ - π），
我们认为新的边缘大致和原来的边缘平行，我们获得了一个笔画的切片。记录该笔画的宽度，并把我们刚刚遍历的在这条线上的所有像素设为该值。
4. 对于可能属于多条线的像素点，把所有像素的值设为线宽的中值。
5. 使用union-find(互斥集合)数据结构连接重叠的切线。在一个集合里面包括了所有重叠的笔画，每一个集合很可能是一个字母或字符
6. 智能过滤上述线段集合。消除太小的宽或高的字符。消除太长或者太粗的字符（宽高比），消除太稀疏的字符（直径与线宽比）
7. 使用k-d树查找与笔画相似（基于线宽）的字符对，与大小相似（基于宽高）的字符对做交集。计算两个字符的角度。
8. 沿着相似的方向，使用k-d树查找类似的字符。这些字符可能构成一个单词。
9. 生成一个最终图，包括结构单词
