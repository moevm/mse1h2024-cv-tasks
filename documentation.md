# Зависимости
Для корректрой работы приложения необходимо установить и настроить следующие утилиты. Также убедитесь, что на устройстве достаточно памяти (минимум 12-15 гб).

## GitHub CLI
GitHub CLI требуется для получения токена авторизации и дальнейшей его передачи в запускаемые GitHub Actions. Токен автороизации используется в реализованных GitHub Actions для получения информации о открытых пулл-реквестах и получения их содержимого.

### Установка
Для установки GitHub CLI в Arch Linux воспользуйтесь следующей командой:    
```sh
sudo pamac install github-cli
```

Для установки GitHub CLI в Ubuntu Linux воспользуйтесь следующей командой:  
```sh
sudo mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

[Оффициальная документация по установке GitHub CLI.](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)

Проверьте корректность установки с помощью команды:  
```sh
gh --version
```

Пример корректного отклика:  
```sh
gh version 2.45.0 (2024-03-04)
https://github.com/cli/cli/releases/tag/v2.45.0
```

### Настройка
Для дальнейшей работы с GitHub CLI необходимо провести аутентификацию. Для прохождения аутентификации в GitHub CLI воспользуйтесь следующей командой:  
```sh
gh auth login
```
Пример авторизации:  
> ? What account do you want to log into? GitHub.com  
> ? What is your preferred protocol for Git operations on this host? HTTPS  
> ? How would you like to authenticate GitHub CLI? Login with a web browser  
>   
> ! First copy your one-time code: CE09-717A  
> Press Enter to open github.com in your browser...  

Удостоверьтесь в корректном прохождении аутентификации:

```sh
gh auth token
```

В результате выполения команды выше Вы должны получить токен аутентификации.

> #### Примечание
> Если Вам никак не удается установить GitHub CLI или авторизироваться, то Вы можете создать токен аутентификации вручную. Для этого перейдите по следующей [ссылке](https://github.com/settings/tokens). В правом верхнем углу выберите `Generate new token`. При генерации токена установите для него доступ к репозиториям. Сохраните полученный токен.

## SSH
Приложение использует SSH для получения содержимого пулл-реквестов.

> #### Примечание
> В процессе разработки принято решение использовать rsa SSH-ключи, так как данные ключи поддерживаются даже на устаревших системах. Так как приложение работает с использованием GitHub Actions, оно не имеет возможности напрямую получить доступ ко всем SSH ключам, используемым в системе, так как по изначальной задумке GitHub Actions должны запускаться не локально, а на удаленной виртуальной системе. Поэтому указание SSH ключей происходит напрямую при запуске рабочего процесса GitHub Actions, и это должны быть именно rsa ключи.

### Генерация ключей
Для генерации rsa ключей используйте следующую команду, также запомните ключ-пароль (passphrase):  
```sh
cd ~/.ssh
ssh-keygen -t rsa
```
Далее укажите файл в котором будут сохранены ключи (рекомендуемое имя файла - id_rsa).
Для проверки сгенерирванных ключей проверьте содержимое файлов `~/.ssh/id_rsa.pub` (публичный ключ) и `~/.ssh/id_rsa` (приватный ключ), например с помощью команды cat.

### Добавление ключей в GitHub аккаунт
Перейдите по [ссылке](https://github.com/settings/keys). В правом верхнем углу выберите `New SSH key`. В поле `Key` укажите содержимое файла `~/.ssh/id_rsa.pub`.

Для проверки наличия SSH-соединения воспользуйтесь следующей командой, при необходимости введите ключ-пароль(passphrase):
```sh
ssh -T git@github.com
```

Результат вывода должен быть следующим:
```
Hi <Ваше имя пользователя>! You've successfully authenticated, but GitHub does not provide shell access.
```

[Подробная информация о установке SSH-соединения с GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection).

## Act
Утилита Act используется для запуска GitHub Actions на локальной машине.

> #### Примечание
> Обычно данная утилита используется для тестирования GitHub Actions. Вообще запуск GitHub Actions локально это очень странное решение.

### Установка
Для установки Act в Arch Linux воспользуйтесь следующей командой:
```sh
sudo pamac install act
```
> #### Примечание
> Для сборки Act необходимо установить [go](https://go.dev/doc/install).   
> скачайте папку с официального сайта, разархивируйте, зайдите в папку (пример названия папки: go1.22.2.linux-amd64), выполните команды (в данной папке):
> ```sh
> sudo mv go /usr/local
> export PATH=$PATH:/usr/local/go/bin
> go version
> ```
> Также необходимо установить утилиту make:
> ```sh 
> sudo apt install make
> ```

Для установки Act в Ubuntu Linux воспользуйтесь следующими командами, [оффициальный репозиторий Act](https://github.com/nektos/act):
```sh
wget -qO act.tar.gz https://github.com/nektos/act/releases/latest/download/act_Linux_x86_64.tar.gz
sudo tar xf act.tar.gz -C /usr/local/bin act
```
Для удаления tar.gz файла:
```sh
rm -rf act.tar.gz
```
Для проверки корректности установки Act воспользуйтесь следующей командой:
```sh
act --version
```

Пример корректного отклика:
```
act version 0.2.60
```

## Создание корректного .secrets
Передача токенов и ключей в Act для запуска GitHub Actions происходит напрямую через указание файла, содержащего секреты. Синтаксис файла .secrets совпадает с синтаксисом файлов, содержащих переменные окружения. В данном файле должны содержаться следующие переменные: `GITHUB_TOKEN`, `PUBLIC_SSH_KEY`, `PRIVATE_SSH_KEY`.  
Переменная `GITHUB_TOKEN` должна содержать вывод команды `gh auth token` или токен авторизации, созданный Вами вручную.  
Переменная `PUBLIC_SSH_KEY` должна содержать содержимое файла `~/.ssh/id_rsa.pub`.  
Переменная `PRIVATE_SSH_KEY` должна содержать содержимое файла `~/.ssh/id_rsa`, при этом все переносы строк должны быть заменены на последовательность символов `\\n`. Данную замену символов можно сделать командой `cat ~/.ssh/id_rsa | sed ':a;N;$!ba;s/\n/\\\\n/g'`.

Пример содержания корректного .secrets файла:  
```
GITHUB_TOKEN="gho_<Ваш токен аутентификации GitHub>"
PUBLIC_SSH_KEY="ssh-rsa <Ваш публичный ключ SSH rsa>"
PRIVATE_SSH_KEY="-----BEGIN OPENSSH PRIVATE KEY-----\\n<Ваш>\\n<приватный>\\n<ключ>\\n<SSH>\\n<rsa>\\n-----END OPENSSH PRIVATE KEY-----"
```

Для автоматического создания корректного файла .secrets воспользуйтесь скриптом `set_secrets.sh` в корне репозитория. Данный скрипт создаст корректный файл .secrets, находящийся в директории `.gitub/actions`.

## Запуск приложения
Для запуска приложения используйте следующую команду:  
```
act --secret-file .github/actions/.secrets
```

Если Вы создавали файл .secrets вручную, то укажите корректный путь к нему.  

> #### Примечание
> При первом запуске Act, Вам будет предложено выбрать образ для исполнения рабочего прочесса GitHub Actions. GitHub Actions в данном приложении реализованы так, чтобы они могли запускаться даже на Micro образе.

## Создание нового репозитория для работы приложения
Для создания нового репозитория, содержащего GitHub Actions из этого приложения, нажмите на кнопку `Use this template` в правом верхнем углу страницы исходного кода приложения и пройдите стандартную процедуру создания репозитория в GitHub.

# Удачи
cообщения внутри пуллреквестов
