use student;
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25),
SECTION VARCHAR(25), MARKS INT);
insert into STUDENT values('SANIYA', 'DATA SCIENCE', 'A', 90);
insert into STUDENT values('TOM', 'DATA SCIENCE', 'B', 100);
insert into STUDENT values('JACOB', 'DATA SCIENCE', 'B', 95);

select * FROM STUDENT;
