# Настройка и запуск workflow

## Команды для запуска
Перейти в корень репозитория  
Создать необходимый для работы файл .secrets  с помощью команды
`cd $PATH_TO_set_secrets.sh && bash set_secrets.sh`

Установить act, если он не установлен __не через snap__. https://github.com/nektos/act

Запустить программу  
`sudo act --secret-file .github/actions/.secrets`
можно выбирать micro

## Решение ошибок при запуске
Установить gh, если он не установлен __не через snap__. 
Пример установки:
`apt-get update && apt-get install -y curl gpg && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && apt-get update &&  apt-get install -y gh`
Для работы программы используются ssh rsa ключи.  
Генерация ключей на локальной машине:  sudo apt  install gh
`ssh-keygen -t rsa`

Создайте ключ на профиле гитхаб:  
`settings > ssh and gpk keys > new ssh key > <вставьте содержимое файла ~/.ssh/id_rsa.pub>`

Авторизуйтесь в гитхаб cli  
`gh auth login`

Проверка всех необходимых ключей на машине  

`cat ~/.ssh/id_rsa`
`cat ~/.ssh/id_rsa.pub`  
`gh auth token`
