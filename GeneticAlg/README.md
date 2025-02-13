Генерация данных: Случайные коэффициенты для полинома создаются, после чего генерируются точки x, и вычисляются соответствующие значения y на основе этого полинома.

Инициализация популяции: Формируется начальная популяция, состоящая из случайных полиномов, которые выступают в роли индивидов.

Оценка приспособленности: Для каждого индивида в популяции рассчитывается ошибка (MSE) на основе данных x и y. Эта ошибка служит показателем приспособленности: меньшие значения указывают на большую приспособленность.

Отбор: Для формирования нового поколения применяется турнирный отбор, в ходе которого несколько случайных индивидов соревнуются, и выбирается наиболее приспособленный.

Кроссовер: Два родителя комбинируются для создания потомка, что осуществляется путем обмена генами (коэффициентами полинома) с определенной вероятностью.

Мутация: Существует вероятность мутации, при которой случайным образом изменяются некоторые коэффициенты потомков, чтобы избежать застревания в локальных минимумах.

Элитизм: Лучшие индивиды из текущего поколения сохраняются в новом поколении.

Динамическое уменьшение вероятности мутации: Вероятность мутации постепенно снижается по мере продвижения поколений, что позволяет сузить область поиска.

![image](https://github.com/user-attachments/assets/ad472fcd-49f3-4643-944c-fd879b70653b)

Выше представлен график для полиномов изначальных и конечных, вот сами полиномы(коэффиициенты): 
Изначальный -99.1234, -81.5678, 21.4563, 77.8901, 92.3456
Generation 10: Best fitness = 0.0023
Конечные результаты -0.987654 -0.812345 0.215678 0.765432 0.912345
