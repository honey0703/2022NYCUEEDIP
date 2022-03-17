//
//  main.cpp
//  Resolution
//
//  Created by 趙宇涵 on 2022/3/15.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
using namespace std;

//********** Declare parameters **********//
char* _header;
char* _image;

int _width;
int _height;
int _channel;
int _compress;
short _bitDepth;
int _padding;
int _padding_input;
int _padding_output;

char* _header_out;
char* _image_out;
char* _image_out2;
//***************************************//

void clone()
{
    _header_out = _header;
    _image_out = _image;
    _padding_output = _padding_input;
}

bool read(const char *filename)
{
    ifstream file_img(filename, ios::in|ios::binary);

    //read header
    _header = new char[54];  // creat bmp head size
    file_img.read( (char*)_header, sizeof(char)*54 ); // read file_img head data
    _width = *(int*)&_header[18];   // the 18th is width
    _height = *(int*)&_header[22];  // the 22nd is height
    _bitDepth = *(int*)&_header[28];
    _channel = _bitDepth/8;
    _padding_input = (_channel-_width*_channel%4)%4;  // each row need to be 4x size, the rest need to padding as 0.
    cout << "_padding_input: "  << _padding_input << endl;  // print out padding number to check

    _image = new char[ _height*( _width*_channel +_padding_input ) ]; // build an image size to store image
    file_img.read( _image, sizeof(char)*_height*( _width+_padding_input )*_channel );

    file_img.close();
    return true;
}

bool write(const char *filename)
{

    ofstream file_img(filename, ios::out | ios::binary);
    file_img.write( _header_out, sizeof(char)*54);  // write in file_img head data
    file_img.write( _image_out,  sizeof(char)* (((*(int*)&_header_out[18]) + _padding_output) * _channel * (*(int*)&_header_out[22])  )); // write in image data
    file_img.close();

    return true;
}

void quanti(const int quanti_factor){

    _header_out = _header;
    _padding_output = _padding_input;
    _image_out = new char[ ((*(int*)&_header_out[18])*_channel+_padding_output) * (*(int*)&_header_out[22]) ];
    for(int i=0 ; i < ( (*(int*)&_header_out[18])*_channel+_padding_output) * (*(int*)&_header_out[22]) ; ++i)
    {
        _image_out[i] =  ((unsigned int)(_image[i] >> quanti_factor )) << quanti_factor;
    }
}

using namespace std;
int main()
{
    // First QR
    read("/Users/zhaoyuhan/Desktop/input1.bmp");
    quanti(2);
    write("/Users/zhaoyuhan/Desktop/output_data/output1_1.bmp");

    // Second QR
    read("/Users/zhaoyuhan/Desktop/input1.bmp");
    quanti(5);
    write("/Users/zhaoyuhan/Desktop/output_data/output1_2.bmp");

    // Third QR
    read("/Users/zhaoyuhan/Desktop/input1.bmp");
    quanti(7);
    write("/Users/zhaoyuhan/Desktop/output_data/output1_3.bmp");
    return 0;
}
