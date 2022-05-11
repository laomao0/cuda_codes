#include <iostream>
#include <string>
using namespace std;

class Demo{
public:
    Demo(string s);
    ~Demo();
private:
    string m_s;
};
Demo::Demo(string s): m_s(s){ }
Demo::~Demo(){ cout<<m_s<<endl; }

void func(){
    //局部对象
    Demo obj1("1");
}

//全局对象
Demo obj2("2");

int main(){
    //局部对象
    Demo obj3("3");
    //new创建的对象
    Demo *pobj4 = new Demo("4");
    func();
    cout<<"main"<<endl;
  
    return 0;
}