function filepath = unaryFeatureDir(num)

switch num
    case 1
        filepath = './Dataset/images_RGB';
    case 2
        filepath = './Dataset/images_LUV';
    case 3
        filepath = './Dataset/images_RGBGMM';
    case 4
        filepath = './Dataset/images_LUVGMM';
    case 5
        filepath = './Dataset/images_RGBdistFG';
    case 6
        filepath = './Dataset/images_RGBdistBG';
    case 7
        filepath = './Dataset/images_LUVdistFG';
    case 8
        filepath = './Dataset/images_LUVdistBG';
    case 9
        filepath = './Dataset/images_RGBGMMdistFG';
    case 10
        filepath = './Dataset/images_RGBGMMdistBG';
    case 11
        filepath = './Dataset/images_LUVGMMdistFG';
    case 12
        filepath = './Dataset/images_LUVGMMdistBG';
    case 13
        filepath = './Dataset/images_EuclidianFG';
    case 14
        filepath = './Dataset/images_EuclidianBG';
    otherwise
        error('This feature do not exist!')
end
end