# Настройка и запуск workflow

## Команды для запуска
Перейти в корень репозитория  
Создать необходимый для работы файл .secrets  
`bash set_secrets.sh`

Запустить программу  
`sudo act --secret-file .github/actions/.secrets`

## Решение ошибок при запуске
Для работы программы используются ssh rsa ключи.  
Генерация ключей на локальной машине:  
`ssh-keygen rsa`

Создайте ключ на профиле гитхаб:  
`settings > ssh and gpk keys > new ssh key > <вставьте содержимое файла ~/ssh/id_rsa.pub>`

Авторизуйтесь в гитхаб cli  
`gh auth login`

Проверка всех необходимых ключей на машине  

`cat ~/.ssh/id_rsa`  
`cat ~/.ssh/id_rsa.pub`  
`gh auth token`
