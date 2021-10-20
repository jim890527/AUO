#include <iostream>
#include <string>
using namespace std;

class Node{		//link list
	private:
		int data;
		Node *next;
	public:
		Node(){};
		/*Node(int val){create(val);};
		Node* create(int value){
			Node *head = new Node;
			Node *current = head;
			current->data = value;
			current->next = NULL;
			return head;	//use to find head
		}*/
		Node* input(Node *current, int inp){	//insert new node in list
			current->next = new Node;
			current = current -> next;	//direction	to new node	
			printf("%d\n",current);
			current->data = inp;
			current->next = NULL;
			return current;
		}
		void insert(Node *x,Node *y,int inp){	//insert y after x
			y->data = inp; 
			y->next = x->next;
			x->next = y;
		}
		void del(Node *x,Node *y){	//want delete y, assume x is in front of y
			if(x==NULL){
				Node *head = new Node;
				head = y->next;	//if y is head then x is null so y->next become head
			}	
			else{
				x->next = y->next;
			}
		}
		void list(Node *x){		//print list
			do{
				printf("%d\n",x->data);
				x=x->next;
				//printf("%d\n",x);
			}while(x != NULL);
		}
		int output(Node *x){	//output node x's data
			int outp = x->data;
			//x = x->next;
			return outp;
		}
};

class Binary_tree{	//BST use inorder can find data From small to large
	private:
		int data,n=0,HL=0,HR=0;
		Binary_tree *left,*right;
	public:
		Binary_tree(){};
		Binary_tree *root(int inp){
			Binary_tree *R;
			R->data = inp;
			return R;	//root node address
		}
		void insert(Binary_tree *node, int inp){	//insert data in binary search tree (node->right's data must > node's data, vice versa)
			if(inp > node->data){	//if data > node->data then insert data to right node
				do{
					if(node->right == NULL){	//if not is leaf then Down growth
						node->right = new Binary_tree;
						node = node->right;
						node->data = inp;
						//printf("R");
						break;	//break this loop
					}
					else{
						 node = node->right;
						 insert(node,inp);	//need condition again because go to the next node
					}
				}while(node!=NULL);
			}
			else if(inp < node->data){	//if data < node->data then insert data to left node
				do{
					if(node->left == NULL){		//if not is leaf then Down growth
						node->left = new Binary_tree;
						node = node->left;
						node->data = inp;
						//printf("L");
						break;	//break this loop
					}
					else{
						 node = node->left;
						 insert(node,inp);	//need condition again because go to the next node
					}
				}while(node!=NULL);
			}
			else printf("ERROR!!");	//ERROR!!
		}
		void preorder(Binary_tree *node){
			if(node != NULL){
				printf("%d",node->data);//	print D
				preorder(node->left);	//	print L
				preorder(node->right);	//	print R
			}
		}
		void inorder(Binary_tree *node){	//use this can find data From small to large in BST
			if(node != NULL){
				preorder(node->left);	//	print L
				printf("%d",node->data);//	print D
				preorder(node->right);	//	print R
			}
		}
		void postorder(Binary_tree *node){
			if(node != NULL){
				preorder(node->left);	//	print L
				preorder(node->right);	//	print R
				printf("%d",node->data);//	print D
			}
		}
		int count(Binary_tree *node){	//count node
			if(node != NULL){
				n++;	//root = first node
				count(node->left);		//left node
				count(node->right);		//right node
				return n;
			}
			else return 0;
		}
		int height(Binary_tree *node){	//count height
			if(node != NULL){
				HL = height(node->left);	//left height
				HR = height(node->right);	//right height
				if(HL<HR)HL=HR;		//max
				return HL+1;	// + root level
			}
			else return 0;
		}
		void swap(Binary_tree *node){	//exchange between left node and right node in Tree
			if(node != NULL){
				swap(node->left);	//run all node
				swap(node->right);	//run all node
				Binary_tree *temp = node->left;
				node->left = node->right;
				node->right = temp;
			}
		}
};

class Stack_link{	//use link list to make stack
	private:
		int data,size,count;
		Stack_link *next;
		Stack_link *ntop;
	public:
		Stack_link(){ntop = NULL;};
		Stack_link(int n){create(n);};	//constructor
		void create(int n){
			size = n;
			count = 0;
			ntop = NULL;
		}
		bool isfull(){
			if(count==size)
				return true;
			else
				return false;
		}
		bool isempty(){
			if(ntop==NULL)
				return true;
			else
				return false;
		}
		void push(int a){		
			push((char)a);
		}
		void push(char a){
			if(isfull()){
				printf("IsFull\r\n");
			}
			else{
				Stack_link *t = new Stack_link;
				t->data = a;
				t->next = ntop;
				ntop = t;
				count++;	
			}		
		}
		char pop(){
			if(isempty()){
				printf("IsEmpty");
				return 0;
			}
			else{
				int item = ntop->data;
				ntop = ntop->next;
				return item;
			}		
		}
		char Top(){
			if(isempty())
				return 0;
			else
				return ntop->data;
		}
};

class Stack{
	private:
		int top,size;
		char stack[];
	public:
		Stack(){};
		Stack(int n){create(n);};	//constructor
		void create(int n){
			size = n;
			stack[size];
			top=-1;
		}
		bool isfull(){
			if(top==size-1)
				return true;
			else
				return false;
		}
		bool isempty(){
			if(top==-1)
				return true;
			else
				return false;
		}
		void push(int a){		
			push((char)a);
		}
		void push(char a){
			if(isfull())
				printf("IsFull\r\n");
			else
				stack[++top]=a;
		}
		char pop(){
			if(isempty()){
				printf("IsEmpty");
				return 0;
			}
			else
				return stack[top--];
		}
		char Top(){
			if(isempty())
				return 0;
			else
				return stack[top];
		}
		void print(){
			for(int i=size-1 ; i>=0; i--)printf("%c",stack[i]);
		}
};

class Queue{	//bad queue
	private:
		int front,rear,size;
		int queue[];	
	public:
		Queue(){};
		Queue(int n){create(n);};	//constructor
		void create(int n){
			size = n;
			queue[size];
			front = -1;
			rear = -1;
		}
		bool isfull(){
			if(rear==size-1)
				return true;
			else
				return false;
		}
		bool isempty(){
			if(front==rear)
				return true;
			else
				return false;
		}
		void add(int a){
			if(isfull())
				printf("IsFull\r\n");
			else
				queue[++rear]=a;
		}
		int del(){
			if(isempty()){
				printf("IsEmpty");
				return 0;
			}				
			else
				return queue[++front];				
		}
};

class CQueue{	//CQueue[n] just can storage n-1 elements
	private:
		int front,rear,size,newrear;
		int cqueue[];
	public:
		CQueue(){};
		CQueue(int n){create(n);};	//constructor
		void create(int n){
			size = n;
			cqueue[size];
			front = 0;
			rear = 0;
		}
		bool isfull(){
			if(newrear==front)
				return true;
			else
				return false;
		}
		bool isempty(){
			if(front==rear)
				return true;
			else
				return false;
		}
		void add(int a){
			newrear = (rear+1)%size;	//use to condition, avoid change rear
			if(isfull())
				printf("IsFull\r\n");
			else
				cqueue[rear=newrear]=a;
		}
		int del(){
			if(isempty()){
				printf("IsEmpty");
				return 0;
			}				
			else
				return cqueue[++front%size];				
		}
};

class Adv_CQueue{	//have poor performance than CQueue because need more condition, but can storage sizes elements
	private:
		int front,rear,size;
		int cqueue[];
		bool tag=false;
	public:
		Adv_CQueue(){};
		Adv_CQueue(int n){create(n);};	//constructor
		void create(int n){
			size = n;
			cqueue[size];
			front = 0;
			rear = 0;
		}
		bool isfull(){
			if(rear==front && tag)
				return true;
			else
				return false;
		}
		bool isempty(){
			if(front==rear && !tag)		//tag=F
				return true;
			else
				return false;
		}
		void add(int a){
			if(isfull())
				printf("IsFull\r\n");
			else{
				rear = (rear+1)%size;
				cqueue[rear]=a;
				printf("R=%d,%d\r\n",rear,cqueue[rear]);
				if(rear==front)tag=true;
			}			
		}
		int del(){
			if(isempty()){
				printf("IsEmpty");
				return 0;
			}				
			else{
				front = (front+1)%size;
				printf("F=%d,%d\r\n",front,cqueue[front]);
				if(front==rear)tag=false;
				return cqueue[front];
			}								
		}
		void print(){
			for (int i=0;i<size;i++){
				printf("%d,",cqueue[i]);
			}
		}
};

class Search{	//Binary_search find unsorted list have error!!!
	private:
		
	public:
		Search(){};		//constructor
		int non_sential(int a[], int n, int find){	// n = array.size
			int i=0;
			while(i<n){
				if(a[i]==find)return i;
				else i++;
			}
			//for(int i=0;i<n;i++)if(a[i]==find)return i;
			return 0;	// not found
		}
		int sential(int a[], int n, int find){		// efficacy better than non_sential because not have 'if' condition
			int i=n-1;	// Back to front 
			int tmp = a[0];	// use to recover
			a[0]=find;
			while(a[i]!=find){	// a[0] must is find
				i--;
			}
			a[0]=tmp;	// recover
			return i;	// if return 0 then means not found
		}
		int binary_search(int a[], int find, int f, int r){		//Binary_search find unsorted list have error!!!  //array,find,front,rear
			//printf("(%d,%d)",f,r);
			if(f<=r){
				int mid = (f+r)/2;
				if(a[mid]==find)return mid;
				else if(a[mid]>find)binary_search(a,find,f,mid-1);	//because array is sorted
				else binary_search(a,find,mid+1,r);
			}
			else 
				return 0;	// not found
		}
};

class Sort{		// Small to large sort
	private:
		//int data = [89, 34, 23, 78, 67, 100, 66, 29, 79, 55, 78, 88, 92, 96, 96, 23];
	public:
		//Basic sort
		void insert(int data[], int key, int i){	//back to front compare //Make 'key' compare to others  
			while(key < data[i] && i>=0){	//if want large to small sort then need 'i>=0' condition because 'i' may <0
				data[i+1] = data[i];	// If the key is smaller, the key gives location // data[i+1] = key's location 
				i--;	//back to front compare
			}
			data[i+1] = key;	//if not enter the while loop represent data[i] < key so data[i+1]'s location is key
		}
		void insertsort(int data[], int size){	//back to front compare
			for(int i=1;i<size;i++)insert(data,data[i],i-1);	//i=3 data[3],2
		}
		void selectionsort(int data[], int size){	
			for(int i=0;i<size-1;i++){	//run size-1 times
				int m=i;	//assume 'i' is minimum location
				for(int j=i+1;j<size;j++){	
					if(data[j]<data[m])m=j;	// find minimum/max location
				}
				if(m!=i)swap(data[m],data[i]);	// put smallest to 'i' location
			}
		}
		void bubblesort(int data[], int size){
			for(int i=size-1;i>=0;i--){	//run size times //i must is size-1 because will be use [j+1]
				bool tag = false;
				for(int j=0;j<i;j++){
					if(data[j]>data[j+1]){	//at first minimum/max to rear 
						swap(data[j],data[j+1]);
						tag = true;
					}
				}
				if(!tag)break;	//if not need any swap represent data is sorted
			}
		}
		//Advanced sort
		void quicksort(int data[], int left, int right){	//j=key's destination //quicksort(data, 0, len(data)-1);
		    if(left < right){
		    	int i=left,j=right,key=data[left];	//assume first data is key
		    	do{
		    		while(data[i] <= key)i++;	//find > key, use '<=' because key = data[left]
		    		while(data[j] > key)j--;	//find < key, at last j = key's destination 
		    		if(i < j)swap(data[i],data[j]);		//let > key in key's right, < key in left
				}while(i < j);	// until i,j interlace
				//printf("i=%d,%d,j=%d,%d\n",i,data[i],j,data[j]);
				swap(data[left],data[j]);		//swap left to destination
				quicksort(data, left, j-1);		//do key's left
		    	quicksort(data, j+1, right);	//do key's right
			}
		}
		void merge(int data[],int low,int mid,int high){
			int size = sizeof(data);
		    int arr1[size/2],arr2[size/2];	//Need at most storage 'size/2' elements
		    int n1,n2,i,j;
		    //printf("%d,%d,%d\n",low,mid,high);
		    n1=mid-low+1;	// count mid's left data, include mid
		    n2=high-mid;	// count mid's right data
		    //printf("%d,%d\n",n1,n2);
		    for(i=0; i < n1; i++){	//at last 'low+i' is mid's left divide
		    	//printf("i=%d,%d\n",low+i,data[low+i]);
		        arr1[i]=data[low+i];	//storage mid's left data
		    }
		    for(j=0; j < n2; j++){	//at last 'mid+j+1' is mid's right divide
		    	//printf("j=%d,%d\n",mid+j+1,data[mid+j+1]);
		        arr2[j]=data[mid+j+1];	//storage mid's right data
		    }
		    arr1[i]=9999;	// use to end
		    arr2[j]=9999;	// use to end
		    i=0;
		    j=0;
		    for(int k=low;k<=high;k++){		// Input data to destination
		        if(arr1[i]<=arr2[j]){		// at last arr will be sorted list
					data[k]=arr1[i++];
				}
		        else{
		        	data[k]=arr2[j++];
				}
		    }
		}
		void mergesort(int data[],int low,int high){
		    int mid;
		    if(low < high){
		        mid=(low+high)/2;
		        mergesort(data,low,mid);	// left
		        mergesort(data,mid+1,high);	// right
		        merge(data,low,mid,high);	// sort
		    }
		}
		void maxheap(int data[], int root, int size){	// create max heap
			int maxNode = root;			// Assume root is maximum
			int left  = root*2 + 1;	    // Root's left child
		    int right = root*2 + 2;     // Root's right child
		    // find the maximum
		    if(left < size && (data[left] > data[maxNode]))		// use 'left < size ' avoidance run to sorted array
		        maxNode = left;
		    if(right < size && (data[right] > data[maxNode]))	// use 'right < size' avoidance run to sorted array
		        maxNode = right;
		    // if root is not the maximum
		    if(maxNode != root){
		    	swap(data[maxNode], data[root]);	// Change maximum to root
		    	maxheap(data, maxNode, size);		// Until parent > child
			}
		}
		void heapsort(int data[],int size){
			// Create max heap
			for(int i = (size/2)-1; i >= 0; i--){	// Start from 'size/2' is enough 
		        maxheap(data, i, size);		// at last every parent > children
		    }
		    // sort
		    for(int i = size-1; i > 0; i--){	//Every time find a maximum to the array end
		        swap(data[0],data[i]);	// Move the maximum element at root to the end
		        maxheap(data, 0, i);	// Uncertain is max heap, because above use swap, but can find maximum in root
		    }
		}
};

class Avl_tree{		// AVL_Tree must is ' 1 >= L-R >= -1 '
	// https://www.guru99.com/avl-tree.html#5
	private:
		int d;			// data
		Avl_tree *l;	// L-child
		Avl_tree *r;	// R-child
	public:
		int height(Avl_tree *t) {	// calculation tree's height
			int h = 0;
			if(t != NULL){
		 		int l_height = height(t->l);
		    	int r_height = height(t->r);
		    	int max_height = max(l_height, r_height);
				h = max_height + 1;
		    }
		    return h;
		}
		int fib(int a){
			if(a==0)return 0;
			else if(a==1)return 1;
			else return fib(a-2) + fib(a-1);
		}
		int least(int h){		// AVL least node = Fib(h+2)-1
			return fib(h+2)-1;
		}
		int difference(Avl_tree *t){	// AVL_Tree must is ' 1 >= L-R >= -1 '
			int l_height = height(t->l);
		    int r_height = height(t->r);
		    int b_factor = l_height - r_height;		// L-R
			return b_factor;
		}
		Avl_tree* ll_rotat(Avl_tree *parent){	// Rotate right 
			Avl_tree *t;
			t = parent->l;		// t = parent's left 'subtree'
			parent->l = t->r;	// t->r give to parent
			t->r = parent;		// parent to t->r
			cout<<"Left-Left Rotation";
			return t;
		}
		Avl_tree* rr_rotat(Avl_tree *parent){	// Rotate left 
			Avl_tree *t;
			t = parent->r;		// t = parent's right 'subtree'
			parent->r = t->l;	// t->l give to parent
			t->l = parent;		// parent to t->l
			cout<<"Right-Right Rotation";
			return t;
		}
		Avl_tree* lr_rotat(Avl_tree *parent){	// Rotate to the left and then to the right
			Avl_tree *t;
			rr_rotat(parent->l);	// Rotate to left 
			t = ll_rotat(parent);	// Rotate to right 
			cout<<"Left-Right Rotation";
			return t;
		}
		Avl_tree* rl_rotat(Avl_tree *parent){	// Rotate to the right and then to the left 
			Avl_tree *t;
			ll_rotat(parent->r);	// Rotate to right 
			t = rr_rotat(parent);	// Rotate to left 
			cout<<"Right-Left Rotation";
			return t;
		}
		Avl_tree* balance(Avl_tree* t){		// Let tree to balance
			int bal_factor = difference(t);
			if(bal_factor > 1){				// Left is too high
				if(difference(t->l) > 0)	// if '> 0' represent L > R
			        t = ll_rotat(t);
			    else
			        t = lr_rotat(t);
			}
			else if(bal_factor < -1){		// right is too high
				if(difference(t->r) > 0)	// if '> 0' represent L > R
			        t = rl_rotat(t);
			    else
			       	t = rr_rotat(t);
			}
			return t;
		}
		Avl_tree* insert(Avl_tree *r, int data){
			if(r == NULL){		// node is NULL
				r = new Avl_tree;
			    r->d = data;
			    r->l = NULL;
			    r->r = NULL;
			    return r;
			}
			else if(data < r->d){		// insert to node->left
			    r->l = insert(r->l, data);
			    r = balance(r);
			}
			else if(data >= r->d){		// insert to node->right
			    r->r = insert(r->r, data);
			    r = balance(r);
			}
			return r;
		}
		void show(Avl_tree *p, int l){
			int i;
			if(p != NULL){
			    show(p->r, l + 1);
			    cout<<" ";
			    if(p == r)
			        cout << "Root -> ";
			    for(i = 0; i < l && p != r; i++)
			        cout << " ";
			        cout << p->d;
			        show(p->l, l + 1);
			}
		}
		void preorder(Avl_tree *t){
			if(t == NULL) return;
		    cout << t->d << " ";
		    preorder(t->l);
		    preorder(t->r);
		}
		void inorder(Avl_tree *t){
			if(t == NULL) return;
		    inorder(t->l);
		    cout << t->d << " ";
		    inorder(t->r);
		}
		void postorder(Avl_tree *t){
			if(t == NULL) return;
		    postorder(t->l);
		    postorder(t->r);
		    cout << t->d << " ";
		}
};

void stack_queue(int size, int x, int y){
	//static int x=3,y=2; //x input, y output
	printf("Stack:\r\n");
	Stack s1(size);
	for(int i=0;i<x;i++)s1.push(i+1);		//push
	for(int i=0;i<y;i++)printf("%d\r\n",s1.pop());	//pop
	printf("Link_Stack:\r\n");
	Stack_link s2(size);	//use linklist
	s2.push('A');s2.push('B');s2.push('C');s2.push('D');s2.push('E');
	for(int i=0;i<y;i++)printf("%c\r\n",s2.pop());
	printf("\r\nQueue:\r\n");
	Queue q1(size);
	for(int i=0;i<x;i++)q1.add(i+1);	//add
	for(int i=0;i<y;i++)printf("%d\r\n",q1.del());	//del
	printf("\r\nCQueue:\r\n");
	CQueue cq1(size);
	for(int i=0;i<x;i++)cq1.add(i+1);	//add
	for(int i=0;i<y;i++)printf("%d\r\n",cq1.del());		//del	//just can storage size-1 elements
	printf("\r\nAdv_CQueue:\r\n");
	Adv_CQueue acq1(size);
	for(int i=0;i<x;i++)acq1.add(i+1);	//add
	for(int i=0;i<y;i++)printf("%d\r\n",acq1.del());	//del
}

void postfix(string infix){		//infix change to postfix (if X>top then push(X))
	Stack s1(infix.size());
	char s[infix.size()];
	printf("\nInfix = ");
	for(int i=0 ; i<infix.size(); i++){		//infix
		//cout << infix[i];
		s[i]=infix[i];
		printf("%c",s[i]);
	}
	printf("\nPostfix = ");
	for(int i=0 ; i<infix.size(); i++){		//postfix
		char a;
		if(s[i]=='(')s1.push(s[i]);	//if get '(' then push
		else if(s[i]==')'){		//if get ')' then pop until get '('
			do{
				a=s1.pop();
				if(a!='(')printf("%c",a);
			}while(a!='(');
		}
		else if(s[i]=='*'||s[i]=='/'){		//if get '*' or '/', pop operator until top's level < i (+,-)
			if(s1.Top()=='*'||s1.Top()=='/'){
				do{
					a=s1.pop();
					if(a!='0')printf("%c",a);
				}while(s1.Top()=='*'||s1.Top()=='/');	//pop until not get '*' or '/'
				s1.push(s[i]);
			}
			else{
				s1.push(s[i]);
			}	
		}	
		else if(s[i]=='+'||s[i]=='-'){	//if get '+' or '-', pop operator until top's level < i
			if(s1.Top()=='('||s1.isempty()){
				s1.push(s[i]);
			}
			else{
				do{
					a=s1.pop();
					if(s1.Top()!='('||!s1.isempty())printf("%c",a);
				}while(s1.Top()!='('&&!s1.isempty());	//pop until get '(' or ' ', if use || have infinite loop because get '(' impossible is ' '
				s1.push(s[i]);
			}
		}
		else{	//direct print operand (ex.ABC)
			cout<< infix[i];
		}
	}
	while(!s1.isempty()){	//pop stack until is empty
		printf("%c",s1.pop());
	}
	printf("\n");
}

int calculation(string infix){	//postfix calculation, if is operand direct push it
	/*Stack s1;
	postfix=postfix(infix);		//infix change to postfix
	while(postfix!=''){
		if(postfix==operand)s1.push(postfix);	//if is operand direct push it
		else{	//is operator
			for(int i=0;i<2;i++)s1.pop(); //pop two operand
			//Calculation results //two operand one operator
			s1.push(results);
		}
		//to next postfix[];
	}
	do{
		ans+=s1.pop();
	}until(s1.isempty())
	return ans;*/
}

void linked_list(){
	Node n1;	//n1 is head node
	Node *c = &n1;	//c is current node
	for(int i=1;i<=5;i++)c=n1.input(c,i);	//node,data
	printf("\r\n");
	n1.list(&n1);
}

void tree(){
	Binary_tree b1;
	Binary_tree *r;
	printf("A");
	r=b1.root(100);
	printf("A");
	b1.insert(r,50);
	printf("A");
}

void search(int data[], int find, int size){
	Search S1;
	printf("Binary_search: %d is in Location %d\n",find,S1.binary_search(data,find,0,size));	//array,find,front,rear  //Binary_search find unsorted list have error!!!
	printf("Linear_search: %d is in Location %d\n",find,S1.sential(data,size,find));
	printf("Linear_search: %d is in Location %d\n",find,S1.non_sential(data,size,find));
	//printf("%d",sizeof(arr)/sizeof(arr[0]));	//arr.size
}

void sort(int data[], int size){
	Sort S1;
	printf("Unsorted:");	//Binary_search find unsorted list have error!!!
	for(int i=0;i<size;i++)printf("%d, ",data[i]);
	printf("\n");
	printf("Sorted:");
	S1.heapsort(data,size);
	for(int i=0;i<size;i++)printf("%d, ",data[i]);
	printf("\n");
}

void search_sort(){
	int data[]={88, 34, 23, 78, 67, 100, 66, 29, 79, 55, 89, 92, 96, 78, 40};
	//int data[]={26,5,13,17,33,59,15,61};
	int size = sizeof(data)/sizeof(data[0]);
	int find = 66;
	printf("Binary_search find unsorted list have error!!!\n");
	search(data,find,size);
	sort(data,size);
	search(data,find,size);
}

int main(int argc, char** argv){
	//stack_queue(3,5,4);	//size,input,output	//Adv not ok!!!
	//postfix("1/(2-3*4)+5-6*7");	//6-47
	//postfix("A+B*C-D/E");	//6-43
	//linked_list();
	//tree();		//not ok!!!
	search_sort();
}
