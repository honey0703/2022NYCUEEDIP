//
//  main.cpp
//  Scaling
//
//  Created by 趙宇涵 on 2022/3/15.
//
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>

using namespace std;

//********** Declare parameters **********//
#define SCALE_UP true
#define SCALE_DOWN false

float scaling_factor = 1.5; // change to the scaling rate.

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

void scaling(const bool bl_up_scaling)
{
    if(bl_up_scaling)
    {
        cout << "Scale up!" << endl;
        *(int*)&_header[18] = int(_width*scaling_factor);
        *(int*)&_header[22] = int(_height*scaling_factor);
        _padding_output = (_channel-int(_width*scaling_factor*_channel)%4)%4 ;
        cout << "_padding_output: "  << _padding_output << endl; // print out padding number to check
        _image_out = new char[ int(_height*scaling_factor * (_width*scaling_factor* _channel+_padding_output) ) ];
        
        // ********* Start Bilinear Interpolation ************** //
        double i_in, j_in ;
       for( int i=0 ; i < _height*scaling_factor ; ++i ){
            for( int j=0 ; j < _width*scaling_factor ; ++j ){
                for (int k=0 ; k < _channel ; ++k ){
                    i_in = ( ((double)_height-1)/((double)_height*scaling_factor-1) ) * (double)i ;
                    j_in = ( ((double) _width-1)/((double) _width*scaling_factor-1) ) * (double)j ;

                    _image_out[ i*(int(_width*scaling_factor*_channel)+_padding_input)+j*_channel+k ] =
                    (unsigned char)_image[ ((int)floor(i_in))*(_width*_channel+_padding_input)+((int)floor(j_in))*_channel+k] *( 1-( i_in-floor(i_in) ) )*( 1-( j_in-floor(j_in) ) )
                    + (unsigned char)_image[ ((int) ceil(i_in))*(_width*_channel+_padding_input)+((int)floor(j_in))*_channel+k]
                    *(i_in-floor(i_in))*( 1-( j_in-floor(j_in) ) )
                    + (unsigned char)_image[ ((int)floor(i_in))*(_width*_channel+_padding_input)+((int) ceil(j_in))*_channel+k ]
                    *( 1-( i_in-floor(i_in) ) )*(j_in-floor(j_in))
                    + (unsigned char)_image[ ((int) ceil(i_in))*(_width*_channel+_padding_input)+((int) ceil(j_in))*_channel+k]
                    *(i_in-floor(i_in))*(j_in-floor(j_in));
                    
        // ********************* End *************************** //
                }
            }
        }
        _header_out = _header;
    }
    else
    {
        cout << "Scale down!" << endl;
        *(int*)&_header[18] = int(_width/scaling_factor);
        *(int*)&_header[22] = int(_height/scaling_factor);
        _padding_output = (_channel-int(_width*scaling_factor*_channel)%4)%4 ;
        _image_out = new char[ int(_height*scaling_factor * (_width*scaling_factor* _channel+_padding_output) ) ];
        cout << "_padding_output: "  << _padding_output << endl;
        cout << (*(int*)&_header[18]) << ", " << (*(int*)&_header[22]) << endl;

        for( int i=0 ; i < _height/scaling_factor ; ++i ){
            for( int j=0 ; j < _width/scaling_factor ; ++j ){
                for (int k=0 ; k < _channel ; ++k ){
                    _image_out[ int(i*(_width/scaling_factor*_channel+_padding_output)+j*_channel+k) ] =  _image[ int(i*scaling_factor)*(_width*_channel+_padding_input)+int(j*scaling_factor)*_channel+k ] ;
                }
            }
        }
        _header_out = _header;
    }
}

int main()
{
    read("/Users/zhaoyuhan/Desktop/input2.bmp");  // input image path
    scaling(SCALE_UP);
    write("/Users/zhaoyuhan/Desktop/output_data/output2_up.bmp"); // output image path

    read("/Users/zhaoyuhan/Desktop/input2.bmp");  // input image path
    scaling(SCALE_DOWN);
    write("/Users/zhaoyuhan/Desktop/output_data/output2_down.bmp");  // output image path

    return 0;
}




